# Container for the deployed demo (Hugging Face Spaces, Cloud Run, Fly, or any
# host that runs a plain image and sets $PORT).
#
# Two native dependencies drive the layout:
#   * LightGBM needs OpenMP at runtime -- libgomp1. Without it the classifier
#     fails to load, the API still answers 200, and /api/health reports
#     classifier_loaded: false. The frontend has a degraded-mode banner for
#     exactly this, but it is far better to just install the library.
#   * batman-package compiles from C at install time, so gcc/g++ are needed to
#     build it and useless afterwards. They are purged in the same layer.

FROM python:3.12-slim

# Runtime native libs. Kept in a separate, rarely-changing layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so editing the app does not reinstall ~400 MB of
# scientific Python. Build toolchain is added and removed inside one layer so
# it never reaches the final image.
# scipy, astropy and pandas each ship their own test suites inside the installed
# package -- ~91 MB that is never imported at runtime. The final find calls
# strip those, the bytecode caches and the type stubs. Image size is what a
# scale-from-zero cold start has to pull, so this is not just tidiness.
COPY requirements-runtime.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && pip install --no-cache-dir --upgrade pip setuptools \
 && pip install --no-cache-dir -r requirements-runtime.txt \
 && apt-get purge -y --auto-remove gcc g++ \
 && rm -rf /var/lib/apt/lists/* \
 && find /usr/local/lib/python3.12/site-packages \
      \( -type d -name tests -o -type d -name test -o -type d -name __pycache__ \) \
      -prune -exec rm -rf {} + \
 && find /usr/local/lib/python3.12/site-packages -name "*.pyi" -delete

# Only what the service reads at runtime. scripts/, tests/, the training
# tables and the 47 MB baseline bank are excluded by .dockerignore.
COPY src/                   ./src/
COPY api/                   ./api/
COPY web/                   ./web/
COPY models/classifier.joblib ./models/
COPY data/cache/TIC_*.npz   ./data/cache/

# Run unprivileged. UID 1000 and port 7860 are what Hugging Face Spaces expects;
# other hosts override the port through $PORT. astropy writes a config and cache
# directory on first use, so HOME must be writable -- a read-only HOME turns the
# first MAST fetch into an error rather than a warning.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PORT=7860

EXPOSE 7860

# Single worker on purpose. One analysis peaks near 330 MB and the API already
# runs a 4-thread pool internally; multiple uvicorn workers would multiply the
# resident set without adding throughput on a 2-vCPU box.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
