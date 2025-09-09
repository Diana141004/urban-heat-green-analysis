#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_delta_lst.py — Delta LST (AFTER - BEFORE), fără poligoane.
Intrări: două GeoTIFF LST aliniabile (Kelvin sau Celsius).
Ieșiri: GeoTIFF cu ΔLST (aceeași unitate ca intrările) + PNG opțional.
"""

import argparse, os, numpy as np, rasterio
from rasterio.warp import reproject, Resampling

def read1(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float32")
        nod = ds.nodata
        prof = ds.profile
    if nod is not None:
        a = np.where(a == nod, np.nan, a)
    return a, prof

def to_ref_grid(ref_prof, arr, prof):
    same = (
        prof["crs"] == ref_prof["crs"] and
        prof["transform"] == ref_prof["transform"] and
        prof["width"] == ref_prof["width"] and
        prof["height"] == ref_prof["height"]
    )
    if same:
        return arr
    dst = np.full((ref_prof["height"], ref_prof["width"]), np.nan, dtype="float32")
    reproject(
        source=arr, destination=dst,
        src_transform=prof["transform"], src_crs=prof["crs"],
        dst_transform=ref_prof["transform"], dst_crs=ref_prof["crs"],
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan
    )
    return dst

def write_tif(path, arr, prof, nodata=np.nan):
    p = prof.copy()
    p.update(driver="GTiff", dtype="float32", count=1, nodata=nodata,
             compress="DEFLATE", tiled=True, blockxsize=512, blockysize=512,
             BIGTIFF="IF_SAFER")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype("float32"), 1)

def save_png(path, arr, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    plt.figure()
    im = plt.imshow(arr, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("ΔLST (°C sau K)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Delta LST (AFTER - BEFORE)")
    ap.add_argument("--before", required=True, help="GeoTIFF LST înainte (ex: 2019)")
    ap.add_argument("--after",  required=True, help="GeoTIFF LST după (ex: 2024)")
    ap.add_argument("--out_tif", required=True, help="Output ΔLST GeoTIFF")
    ap.add_argument("--out_png", help="Output PNG (opțional)")
    ap.add_argument("--png_vmin", type=float, default=None)
    ap.add_argument("--png_vmax", type=float, default=None)
    args = ap.parse_args()

    # Citire BEFORE și AFTER
    before, pb = read1(args.before)
    after,  pa = read1(args.after)

    # Aliniază AFTER pe grila BEFORE (dacă e cazul)
    after_al = to_ref_grid(pb, after, pa)

    # Delta (after - before)
    delta = np.full_like(before, np.nan, dtype="float32")
    m = np.isfinite(before) & np.isfinite(after_al)
    delta[m] = after_al[m] - before[m]

    # Scriere GeoTIFF
    write_tif(args.out_tif, delta, pb)

    # PNG opțional
    if args.out_png:
        save_png(args.out_png, delta, vmin=args.png_vmin, vmax=args.png_vmax)

    # Rezumat
    if np.isfinite(delta).any():
        print(f"ΔLST stats → mean {np.nanmean(delta):.2f}, median {np.nanmedian(delta):.2f}, "
              f"p5 {np.nanpercentile(delta,5):.2f}, p95 {np.nanpercentile(delta,95):.2f}")
    print("Done.")

if __name__ == "__main__":
    main()
