# scripts/green_priority_from_ndvi.py
import argparse, json, os, numpy as np, rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union

def read1(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        nod = ds.nodata
        prof = ds.profile
    if nod is not None:
        arr = np.where(arr == nod, np.nan, arr)
    return arr, prof

def to_ref_grid(ref_prof, arr, prof):
    # dacă NDVI e deja pe grila referinței (ex. din B04), nu reproiectăm
    if (prof["crs"] == ref_prof["crs"] and
        prof["transform"] == ref_prof["transform"] and
        prof["width"] == ref_prof["width"] and
        prof["height"] == ref_prof["height"]):
        return arr
    from rasterio.warp import reproject, Resampling
    dst = np.full((ref_prof["height"], ref_prof["width"]), np.nan, dtype="float32")
    reproject(
        source=arr, destination=dst,
        src_transform=prof["transform"], src_crs=prof["crs"],
        dst_transform=ref_prof["transform"], dst_crs=ref_prof["crs"],
        resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan
    )
    return dst

def normalize_reflectance(a):
    # dacă valorile par în DN scalate (ex. 0..10000), adu-le la 0..1
    finite = np.isfinite(a)
    if not finite.any():
        return a
    p99 = np.nanpercentile(a[finite], 99)
    # euristic: dacă p99 > 2, presupunem scală 0..10000 și împărțim la 10000
    if p99 > 2.0:
        return a / 10000.0
    return a

def compute_ndvi(nir, red):
    num = nir - red
    den = nir + red
    out = np.full_like(nir, np.nan, dtype="float32")
    m = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[m] = num[m] / den[m]
    return out

def compute_ndwi(green, nir):
    # McFeeters NDWI
    num = green - nir
    den = green + nir
    out = np.full_like(green, np.nan, dtype="float32")
    m = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[m] = num[m] / den[m]
    return out

def write_tif(path, arr, prof, nodata=-9999):
    p = prof.copy()
    p.update(driver="GTiff", dtype="float32", count=1, nodata=nodata,
             tiled=True, blockxsize=512, blockysize=512, compress="DEFLATE", BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **p) as dst:
        out = np.where(np.isfinite(arr), arr, nodata).astype("float32")
        dst.write(out, 1)

def main():
    ap = argparse.ArgumentParser(description="Greening Priority din NDVI (S2 B04/B08) + optional water mask (NDWI).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ndvi", help="NDVI GeoTIFF (dacă îl ai deja).")
    src.add_argument("--b04", help="B04 (Red, 10m) GeoTIFF")
    ap.add_argument("--b08", help="B08 (NIR, 10m) GeoTIFF (obligatoriu dacă folosești --b04)")
    ap.add_argument("--b03", help="B03 (Green, 10m) GeoTIFF (opțional pentru NDWI)")
    ap.add_argument("--ndvi_thr", type=float, default=0.25, help="Pixeli cu NDVI < thr devin candidați (default 0.25).")
    ap.add_argument("--ndwi_thr", type=float, default=0.20, help="Dacă ai B03, exclude apa cu NDWI > thr (default 0.20).")
    ap.add_argument("--min_area_m2", type=float, default=3000.0, help="Arie minimă poligon (m²) pt. export GeoJSON.")
    ap.add_argument("--simplify_m", type=float, default=1.5, help="Simplificare poligon (m) pt. GeoJSON.")
    ap.add_argument("--out_raster", required=True, help="Output raster scor (1 - NDVI).")
    ap.add_argument("--out_geojson", required=True, help="Output GeoJSON cu poligoane prioritare.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_raster), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_geojson), exist_ok=True)

    if args.ndvi:
        ndvi, prof = read1(args.ndvi)
    else:
        # citim benzi și calculăm NDVI
        if not args.b08:
            raise SystemExit("Dacă folosești --b04 trebuie să dai și --b08.")
        red,  prof = read1(args.b04)
        nir,  prof8 = read1(args.b08)
        # aducem NIR la grila lui RED (referință)
        nir = to_ref_grid(prof, nir, prof8)
        red = normalize_reflectance(red)
        nir = normalize_reflectance(nir)
        ndvi = compute_ndvi(nir, red)

    # opțional: apă (NDWI) din B03/B08
    water_mask = None
    if args.b03 and args.b08:
        green, pg = read1(args.b03)
        nir2,  pn = read1(args.b08)
        nir2 = to_ref_grid(prof, nir2, pn)
        green = to_ref_grid(prof, green, pg)
        green = normalize_reflectance(green)
        nir2  = normalize_reflectance(nir2)
        ndwi = compute_ndwi(green, nir2)
        water_mask = np.isfinite(ndwi) & (ndwi > args.ndwi_thr)

    # scor = 1 - NDVI (mai mic NDVI => scor mai mare)
    score = np.where(np.isfinite(ndvi), 1.0 - ndvi, np.nan)
    # candidați: NDVI sub prag și nu apă (dacă avem masca apei)
    cand = np.isfinite(ndvi) & (ndvi < args.ndvi_thr)
    if water_mask is not None:
        cand &= ~water_mask

    # scriem raster scor
    write_tif(args.out_raster, score, prof)

    # poligonizare pe candidați (binari)
    if not np.any(cand):
        with open(args.out_geojson, "w", encoding="utf-8") as f:
            json.dump({"type":"FeatureCollection","features":[]}, f)
        print("Niciun pixel candidat sub prag. Gata.")
        return

    # shapes: 1 pentru candidați, 0 altfel
    mask_uint = cand.astype("uint8")
    geoms = []
    for geom, val in shapes(mask_uint, mask=mask_uint==1, transform=prof["transform"]):
        if val != 1:
            continue
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        geoms.append(g)

    merged = unary_union(geoms)
    if merged.is_empty:
        with open(args.out_geojson, "w", encoding="utf-8") as f:
            json.dump({"type":"FeatureCollection","features":[]}, f)
        print("Nimic de exportat (după unire).")
        return

    if merged.geom_type == "Polygon":
        polys = [merged]
    else:
        polys = list(merged.geoms)

    # estimăm aria: dacă CRS-ul e metric (EPSG proiectat), e deja m²
    feats = []
    px_area = abs(prof["transform"].a * prof["transform"].e)  # m² dacă CRS-ul e metric
    for poly in polys:
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        area_m2 = poly.area
        # fallback slab dacă CRS geografic (grade) – tot filtrăm totuși după o limită „m²”
        if area_m2 < args.min_area_m2:
            continue

        simp = poly.simplify(args.simplify_m, preserve_topology=True)

        feats.append({
            "type":"Feature",
            "geometry": mapping(simp),
            "properties":{
                "ndvi_thr": args.ndvi_thr,
                "priority_mean": float(np.nanmean((1.0 - ndvi)[cand])),
                "area_m2": round(float(area_m2), 1)
            }
        })

    fc = {"type":"FeatureCollection","features":feats, "crs":{"type":"name","properties":{"name":str(prof['crs'])}}}
    with open(args.out_geojson, "w", encoding="utf-8") as f:
        json.dump(fc, f)

    print(f"Scris: {args.out_raster}")
    print(f"Scris: {args.out_geojson}  (poligoane: {len(feats)})")

if __name__ == "__main__":
    main()
