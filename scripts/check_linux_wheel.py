#!/usr/bin/env python
"""Verify the LightGBM Linux wheel's OpenMP dependency without a Linux box.

The project is developed on macOS but trained on Colab/Kaggle, so "it works
here" says nothing about whether it works there. The specific risk is
LightGBM's OpenMP runtime: whether `libgomp` ships inside the manylinux wheel
or has to come from the host decides whether `apt-get install libgomp1` is
required or redundant, and the README asserted the wrong answer.

That question is answerable from any platform, because it is a property of the
published artifact rather than of the running machine: download the wheel and
read the ELF dynamic section of its compiled extension. What this CANNOT verify
is a full training run inside a live Colab session -- that still needs someone
to run it there.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import urllib.request
import zipfile


def dt_needed(elf: bytes) -> tuple[list[str], list[str]]:
    """Return (DT_NEEDED entries, RPATH/RUNPATH entries) from an ELF64 image."""
    if elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        raise ValueError("not a little-endian ELF64 object")
    e_shoff, = struct.unpack_from("<Q", elf, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 0x3A)

    def section(i):
        o = e_shoff + i * e_shentsize
        name, typ, flags, addr, off, size, link, info, align, entsize = \
            struct.unpack_from("<IIQQQQIIQQ", elf, o)
        return dict(name=name, off=off, size=size, link=link)

    secs = [section(i) for i in range(e_shnum)]
    shstr = secs[e_shstrndx]

    def name_of(s):
        o = shstr["off"] + s["name"]
        return elf[o:elf.index(b"\0", o)].decode()

    dyn = next(s for s in secs if name_of(s) == ".dynamic")
    dynstr = secs[dyn["link"]]

    def strat(val):
        o = dynstr["off"] + val
        return elf[o:elf.index(b"\0", o)].decode()

    needed, runpath = [], []
    for i in range(dyn["size"] // 16):
        tag, val = struct.unpack_from("<qQ", elf, dyn["off"] + i * 16)
        if tag == 0:
            break
        if tag == 1:
            needed.append(strat(val))
        elif tag in (15, 29):                 # DT_RPATH, DT_RUNPATH
            runpath.append(strat(val))
    return needed, runpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="lightgbm")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    meta = json.load(urllib.request.urlopen(
        f"https://pypi.org/pypi/{args.package}/json", timeout=args.timeout))
    version = meta["info"]["version"]
    wheels = [f for f in meta["releases"][version]
              if "manylinux" in f["filename"] and "x86_64" in f["filename"]]
    if not wheels:
        print(f"no manylinux x86_64 wheel for {args.package} {version}")
        return 1
    w = wheels[0]
    print(f"{args.package} {version}")
    print(f"  wheel: {w['filename']}  ({w['size'] // 1024} KB)")

    z = zipfile.ZipFile(io.BytesIO(
        urllib.request.urlopen(w["url"], timeout=args.timeout).read()))
    sos = [n for n in z.namelist() if n.endswith((".so", ".so.1"))]
    print(f"  shared objects inside the wheel: {sos}")

    ext = next((n for n in sos if "lib_lightgbm" in n or n.endswith(".so")), None)
    if ext is None:
        print("  no compiled extension found")
        return 1
    needed, runpath = dt_needed(z.read(ext))
    print(f"  {ext}")
    print(f"    DT_NEEDED : {'  '.join(needed)}")
    print(f"    DT_RUNPATH: {runpath or '(none)'}")

    omp = [n for n in needed if "gomp" in n or "omp" in n]
    vendored = [s for s in sos if "gomp" in s or "omp" in s]
    print()
    if omp and not vendored:
        print(f"VERDICT: {', '.join(omp)} is an EXTERNAL dependency with no "
              f"vendored copy in the wheel.")
        print("         `apt-get install -y libgomp1` is REQUIRED on a base "
              "image that lacks it.")
        print("         (Colab/Kaggle images normally already carry it -- that "
              "is the image's doing, not the wheel's.)")
    elif vendored:
        print(f"VERDICT: OpenMP is vendored inside the wheel ({vendored}); the "
              f"apt-get line is redundant.")
        print("         README says otherwise -- update it.")
    else:
        print("VERDICT: no OpenMP dependency detected at all; investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
