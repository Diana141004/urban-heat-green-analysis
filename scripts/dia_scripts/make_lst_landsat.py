#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_lst_landsat.py
Calculează LST (Land Surface Temperature) din Landsat 8/9 TIRS Band 10 folosind metoda single-channel (emisivitate din NDVI).
Intrări: B10.tif + (B5.tif, B4.tif) sau ndvi.tif + MTL.txt
Ieșiri: LST (Kelvin) GeoTIFF, opțional LST (Celsius) GeoTIFF, PNG quicklook.
"""

import argparse
import math
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import Affine

# ---- Utilitare ----

def parse_mtl_constants(mtl_path):
    """Extrage ML/AL și K1/K2 pentru BAND_10 din MTL (Landsat 8/9 Collection 2)."""
    keys = {
        "RADIANCE_MULT_BAND_10": None,
        "RADIANCE_ADD_BAND_10": None,
        "K1_CONSTANT_BAND_10": None,
        "K2_CONSTANT_BAND_10": None,
    }
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            for k in list(keys.keys()):
                if line.startswith(k):
                    # Format tipic: KEY = VALUE
                    parts = line.split("=")
                    if len(parts) == 2:
                        try:
                            keys[k] = float(parts[1].strip())
                        except ValueError:
                            pass
    missing = [k for k, v in keys.items() if v is None]
    if missing:
        raise ValueError(f"Nu am găsit în MTL cheile: {missing}")
    return keys["RADIANCE_MULT_BAND_10"], keys["RADIANCE_ADD_BAND_10"], keys["K1_CONSTANT_BAND_10"], keys["K2_CONSTANT_BAND_10"]

def read_align_like(src_path, like_profile):
    """Citește un raster și îl reproiectează/realiniază pe grila 'like_profile' (aceeași rezoluție/transform/proiecție)."""
    with rasterio.open(src_path) as src:
        if (src.crs == like_profile["crs"] and
            src.transform == like_profile["transform"] and
            src.width == like_profile["width"] and
            src.height == like_profile["height"]):
            arr = src.read(1, masked=True).astype("float32")
        else:
            arr = rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=np.empty((like_profile["height"], like_profile["width"]), dtype="float32"),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=like_profile["transform"],
                dst_crs=like_profile["crs"],
                resampling=Resampling.bilinear,
            )[0]
            # Setăm mask simplu (valori NaN acolo unde source era nodata)
            with rasterio.open(src_path) as s2:
                m = s2.read(1, masked=True).mask
            if m.shape != arr.shape:
                # resamplăm masca nearest
                m_res = rasterio.warp.reproject(
                    source=m.astype("uint8"),
                    destination=np.empty((like_profile["height"], like_profile["width"]), dtype="uint8"),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=like_profile["transform"], dst_crs=like_profile["crs"],
                    resampling=Resampling.nearest)[0].astype(bool)
                arr = np.ma.array(arr, mask=m_res)
            else:
                arr = np.ma.array(arr, mask=m)
    return arr

def save_geotiff(path, data, like_profile, dtype="float32", nodata=np.nan):
    profile = like_profile.copy()
    profile.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)

def save_png_quicklook(path, data, vmin=None, vmax=None):
    """Salvează un PNG de previzualizare (negeoreferențiat)."""
    import matplotlib.pyplot as plt
    plt.figure()
    im = plt.imshow(data, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# ---- LST core ----

def compute_ndvi(nir, red):
    nd = (nir - red) / (nir + red + 1e-10)
    nd = np.clip(nd, -1, 1)
    nd = np.ma.array(nd, mask=(nir.mask | red.mask))
    return nd

def compute_emissivity_from_ndvi(ndvi, ndvi_min=-0.2, ndvi_max=0.7):
    fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-10)) ** 2
    fvc = np.clip(fvc, 0.0, 1.0)
    eps = 0.004 * fvc + 0.986
    eps = np.clip(eps, 0.97, 0.999)  # limite rezonabile
    eps = np.ma.array(eps, mask=ndvi.mask)
    return eps

def compute_lst_kelvin(b10_dn, mtl, ndvi=None, red=None, nir=None, ndvi_min=-0.2, ndvi_max=0.7):
    # 1) DN -> Radiance
    ML, AL, K1, K2 = parse_mtl_constants(mtl)
    radiance = ML * b10_dn + AL
    radiance = np.ma.array(radiance, mask=b10_dn.mask | (radiance <= 0))

    # 2) Brightness Temperature (Kelvin)
    bt = K2 / np.log((K1 / (radiance + 1e-10)) + 1.0)

    # 3) Emisivitate din NDVI
    if ndvi is None:
        if red is None or nir is None:
            raise ValueError("Ai nevoie de NDVI (ndvi) sau de RED (B4) + NIR (B5).")
        ndvi = compute_ndvi(nir, red)
    eps = compute_emissivity_from_ndvi(ndvi, ndvi_min, ndvi_max)

    # 4) LST corectat de emisivitate (Kelvin)
    lam = 10.895e-6       # m, TIRS Band 10 efectiv
    c2 = 1.4387769e-2     # m*K, constanta 2 Planck
    lst = bt / (1.0 + (lam * bt / c2) * np.log(eps + 1e-10))

    # mască finală
    mask = b10_dn.mask
    if ndvi is not None:
        mask = mask | ndvi.mask
    lst = np.ma.array(lst, mask=mask)
    return lst

# ---- Main CLI ----

def main():
    p = argparse.ArgumentParser(description="Computează LST din Landsat 8/9 Band 10 (single-channel).")
    p.add_argument("--b10", required=True, help="GeoTIFF Band 10 (DN, Landsat 8/9).")
    p.add_argument("--mtl", required=True, help="Fișierul MTL.txt al scenei.")
    p.add_argument("--red", help="GeoTIFF RED (B4) - dacă nu dai --ndvi.")
    p.add_argument("--nir", help="GeoTIFF NIR (B5) - dacă nu dai --ndvi.")
    p.add_argument("--ndvi", help="GeoTIFF NDVI deja calculat, aliniat la B10 (opțional).")
    p.add_argument("--ndvi_min", type=float, default=-0.2, help="NDVI minim pentru Fv (default -0.2).")
    p.add_argument("--ndvi_max", type=float, default=0.7, help="NDVI maxim pentru Fv (default 0.7).")
    p.add_argument("--out_lst_k", required=True, help="Output GeoTIFF LST (Kelvin).")
    p.add_argument("--out_lst_c", help="Output GeoTIFF LST (Celsius, opțional).")
    p.add_argument("--out_png", help="Quicklook PNG (opțional).")
    p.add_argument("--png_vmin", type=float, default=None, help="vmin pentru PNG (ex. 290).")
    p.add_argument("--png_vmax", type=float, default=None, help="vmax pentru PNG (ex. 320).")
    args = p.parse_args()

    # Deschidem B10 ca "like"
    with rasterio.open(args.b10) as src_b10:
        b10_dn = src_b10.read(1, masked=True).astype("float32")
        like_profile = src_b10.profile

    # Pregătim NDVI (ori din fișier, ori din RED+NIR)
    ndvi_arr = None
    if args.ndvi:
        ndvi_arr = read_align_like(args.ndvi, like_profile)
    else:
        if not (args.red and args.nir):
            raise SystemExit("Dă fie --ndvi, fie --red + --nir.")
        red = read_align_like(args.red, like_profile)
        nir = read_align_like(args.nir, like_profile)
        ndvi_arr = compute_ndvi(nir, red)

    # LST
    lst_k = compute_lst_kelvin(
        b10_dn=b10_dn,
        mtl=args.mtl,
        ndvi=ndvi_arr,
        red=None, nir=None,
        ndvi_min=args.ndvi_min,
        ndvi_max=args.ndvi_max
    )

    # Scriem Kelvin
    save_geotiff(args.out_lst_k, lst_k.filled(np.nan), like_profile, dtype="float32", nodata=np.nan)

    # Celsius (opțional)
    if args.out_lst_c:
        lst_c = lst_k - 273.15
        save_geotiff(args.out_lst_c, lst_c.filled(np.nan), like_profile, dtype="float32", nodata=np.nan)

    # PNG quicklook (opțional)
    if args.out_png:
        data_for_png = (lst_k - 273.15).filled(np.nan)  # mai intuitiv în °C
        save_png_quicklook(args.out_png, data_for_png, vmin=args.png_vmin, vmax=args.png_vmax)

    print("Done.")

if __name__ == "__main__":
    main()
