# scripts/ndvi_change.py  — variantă simplă: DOAR B04/B08 perechi
import argparse, os, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

def read1(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype("float32"); nod = ds.nodata; prof = ds.profile
    if nod is not None: a = np.where(a==nod, np.nan, a)
    return a, prof

def to_ref_grid(ref_prof, arr, prof, method=Resampling.bilinear):
    same = (prof["crs"]==ref_prof["crs"] and prof["transform"]==ref_prof["transform"]
            and prof["width"]==ref_prof["width"] and prof["height"]==ref_prof["height"])
    if same: return arr
    dst = np.full((ref_prof["height"], ref_prof["width"]), np.nan, dtype="float32")
    reproject(arr, dst,
              src_transform=prof["transform"], src_crs=prof["crs"],
              dst_transform=ref_prof["transform"], dst_crs=ref_prof["crs"],
              resampling=method, src_nodata=np.nan, dst_nodata=np.nan)
    return dst

def normalize_reflectance(a):
    finite = np.isfinite(a)
    if not finite.any(): return a
    p99 = np.nanpercentile(a[finite], 99)
    return a/10000.0 if p99>2.0 else a

def ndvi(nir, red):
    num, den = nir - red, nir + red
    out = np.full_like(nir, np.nan, dtype="float32")
    m = np.isfinite(num) & np.isfinite(den) & (den!=0)
    out[m] = num[m]/den[m]
    return out

def polygonize(mask_bool, transform):
    geoms=[]
    for geom,val in shapes(mask_bool.astype("uint8"), mask=mask_bool, transform=transform):
        if val!=1: continue
        s = shape(geom)
        if not s.is_valid: s = s.buffer(0)
        if not s.is_empty: geoms.append(s)
    if not geoms: return []
    merged = unary_union(geoms)
    return [merged] if merged.geom_type=="Polygon" else list(merged.geoms)

def write_tif(path, arr, prof, nod=-9999):
    p=prof.copy()
    p.update(driver="GTiff", dtype="float32", count=1, nodata=nod,
             tiled=True, blockxsize=512, blockysize=512, compress="DEFLATE", BIGTIFF="IF_SAFER")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.where(np.isfinite(arr), arr, nod).astype("float32"), 1)

def main():
    ap = argparse.ArgumentParser(description="ΔNDVI (after-before) din perechi B04/B08.")
    ap.add_argument("--b04_before", required=True, help="Red BEFORE (GeoTIFF)")
    ap.add_argument("--b08_before", required=True, help="NIR BEFORE (GeoTIFF)")
    ap.add_argument("--b04_after",  required=True, help="Red AFTER (GeoTIFF)")
    ap.add_argument("--b08_after",  required=True, help="NIR AFTER (GeoTIFF)")

    ap.add_argument("--loss_thr", type=float, default=0.15, help="ΔNDVI ≤ -loss_thr => loss polygon")
    ap.add_argument("--gain_thr", type=float, default=0.15, help="ΔNDVI ≥ +gain_thr => gain polygon")
    ap.add_argument("--min_area_m2", type=float, default=5000.0, help="min area for polygons")
    ap.add_argument("--simplify_m", type=float, default=2.0, help="geometry simplification")
    ap.add_argument("--out_delta", required=True, help="Output ΔNDVI GeoTIFF")
    ap.add_argument("--out_loss_geojson", help="Output GeoJSON pentru pierdere NDVI")
    ap.add_argument("--out_gain_geojson", help="Output GeoJSON pentru câștig NDVI")
    args = ap.parse_args()

    # BEFORE
    red_b, pb = read1(args.b04_before)
    nir_b, p8b = read1(args.b08_before)
    nir_b = to_ref_grid(pb, nir_b, p8b)
    red_b = normalize_reflectance(red_b); nir_b = normalize_reflectance(nir_b)
    ndvi_b = ndvi(nir_b, red_b)

    # AFTER
    red_a, pa = read1(args.b04_after)
    nir_a, p8a = read1(args.b08_after)
    nir_a = to_ref_grid(pa, nir_a, p8a)
    red_a = normalize_reflectance(red_a); nir_a = normalize_reflectance(nir_a)
    ndvi_a = ndvi(nir_a, red_a)

    # aliniază AFTER la grila BEFORE
    ndvi_a_aligned = to_ref_grid(pb, ndvi_a, pa, method=Resampling.bilinear)

    # ΔNDVI
    delta = np.full_like(ndvi_b, np.nan, dtype="float32")
    m = np.isfinite(ndvi_b) & np.isfinite(ndvi_a_aligned)
    delta[m] = ndvi_a_aligned[m] - ndvi_b[m]
    write_tif(args.out_delta, delta, pb)
    print(f"Wrote ΔNDVI: {args.out_delta}")

    # poligoane opționale
    if args.out_loss_geojson or args.out_gain_geojson:
        import json
        def to_fc(mask, prof, out_path):
            polys = polygonize(mask, prof["transform"])
            feats=[]
            for g in polys:
                if g.is_empty: continue
                if not g.is_valid:
                    g = g.buffer(0)
                    if g.is_empty: continue
                area = g.area
                if area < args.min_area_m2: continue
                simp = g.simplify(args.simplify_m, preserve_topology=True)
                feats.append({"type":"Feature","geometry":mapping(simp),
                              "properties":{"area_m2": round(float(area),1)}})
            fc={"type":"FeatureCollection","features":feats,
                "crs":{"type":"name","properties":{"name":str(prof['crs'])}}}
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path,"w",encoding="utf-8") as f: json.dump(fc,f)
            return len(feats)

        if args.out_loss_geojson:
            loss_mask = np.isfinite(delta) & (delta <= -args.loss_thr)
            nloss = to_fc(loss_mask, pb, args.out_loss_geojson)
            print(f"Loss polygons: {nloss} -> {args.out_loss_geojson}")
        if args.out_gain_geojson:
            gain_mask = np.isfinite(delta) & (delta >=  args.gain_thr)
            ngain = to_fc(gain_mask, pb, args.out_gain_geojson)
            print(f"Gain polygons: {ngain} -> {args.out_gain_geojson}")

    # sumar rapid
    if np.isfinite(delta).any():
        print(f"ΔNDVI stats: mean {np.nanmean(delta):.3f}, median {np.nanmedian(delta):.3f}, "
              f"p5 {np.nanpercentile(delta,5):.3f}, p95 {np.nanpercentile(delta,95):.3f}")

if __name__ == "__main__":
    main()
