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
COPY requirements-runtime.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && pip install --no-cache-dir --upgrade pip setuptools \
 && pip install --no-cache-dir -r requirements-runtime.txt \
 && apt-get purge -y --auto-remove gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Only what the service reads at runtime. scripts/, tests/, the training
# tables and the 47 MB baseline bank are excluded by .dockerignore.
COPY src/                   ./src/
COPY api/                   ./api/
COPY web/                   ./web/
COPY models/classifier.joblib ./models/
COPY data/cache/TIC_*.npz   ./data/cache/

# Hugging Face Spaces runs as UID 1000 and expects the app on 7860. astropy,
# lightkurve and matplotlib all want a writable HOME for their caches; without
# one, astroquery raises rather than warns on the first MAST search.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PORT=7860

EXPOSE 7860

# Single worker on purpose. One analysis peaks near 330 MB and the API already
# runs a 4-thread pool internally; multiple uvicorn workers would multiply the
# resident set without adding throughput on a 2-vCPU box.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
