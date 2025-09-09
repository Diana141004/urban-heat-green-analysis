# scripts/make_priority_polygons.py
import argparse, json, math, numpy as np, rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

def pixel_area_m2(transform):
    return abs(transform.a * transform.e)

def polygonize_mask(mask_bool, transform):
    """Return list of shapely polygons for True cells in mask."""
    geoms = []
    for geom, val in shapes(mask_bool.astype("uint8"), mask=mask_bool, transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if not g.is_empty:
            geoms.append(g)
    return geoms

def main():
    ap = argparse.ArgumentParser(description="Polygonize priority (1-NDVI) > threshold into GeoJSON.")
    ap.add_argument("--priority_tif", required=True, help="GeoTIFF cu 1-NDVI (0..1).")
    ap.add_argument("--thr", type=float, default=0.75, help="Prag pe 1-NDVI (ex. 0.75 ≈ NDVI<0.25).")
    ap.add_argument("--min_area_m2", type=float, default=5000.0, help="Arie minimă poligon (m²).")
    ap.add_argument("--simplify_m", type=float, default=1.5, help="Simplificare geometrie (metri).")
    ap.add_argument("--tile", type=int, default=2048, help="Dimensiunea tile-ului (pixeli) pentru procesare.")
    ap.add_argument("--out", required=True, help="GeoJSON de ieșire.")
    args = ap.parse_args()

    with rasterio.open(args.priority_tif) as ds:
        H, W = ds.height, ds.width
        tr = ds.transform
        crs = ds.crs
        nod = ds.nodata
        px_area = pixel_area_m2(tr)

        # iterăm pe tile-uri ca să evităm blocajul la scene mari
        geoms_all = []
        for r0 in range(0, H, args.tile):
            rh = min(args.tile, H - r0)
            for c0 in range(0, W, args.tile):
                cw = min(args.tile, W - c0)
                win = Window(c0, r0, cw, rh)
                arr = ds.read(1, window=win).astype("float32")

                if nod is not None:
                    arr = np.where(arr == nod, np.nan, arr)

                # masca „prioritară”: 1-NDVI >= thr
                m = np.isfinite(arr) & (arr >= args.thr)
                if not m.any():
                    continue

                # transform local pentru fereastra curentă
                tr_win = rasterio.windows.transform(win, tr)

                # poligonizare locală
                polys = polygonize_mask(m, tr_win)
                if not polys:
                    continue

                geoms_all.extend(polys)

        if not geoms_all:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"type":"FeatureCollection","features":[]}, f)
            print(f"Nu s-a găsit nimic peste pragul {args.thr}. Scris colecție goală: {args.out}")
            return

        # unește zonele atingătoare din toate tile-urile
        merged = unary_union(geoms_all)
        if isinstance(merged, Polygon):
            merged = [merged]
        elif isinstance(merged, MultiPolygon):
            merged = list(merged.geoms)
        else:
            merged = [merged]

        feats = []
        kept = 0
        for g in merged:
            if g.is_empty:
                continue
            if not g.is_valid:
                g = g.buffer(0)
                if g.is_empty:
                    continue
            area_m2 = g.area  # presupunem CRS proiectat (metri). Dacă e geografic, cifra e relativă.
            if area_m2 < args.min_area_m2:
                continue

            simp = g.simplify(args.simplify_m, preserve_topology=True)
            feats.append({
                "type": "Feature",
                "geometry": mapping(simp),
                "properties": {
                    "thr_priority": args.thr,
                    "area_m2": round(float(area_m2), 1)
                }
            })
            kept += 1

        fc = {"type":"FeatureCollection","features":feats,
              "crs":{"type":"name","properties":{"name":str(crs)}}}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(fc, f)

        print(f"Poligoane păstrate: {kept} | Prag (1-NDVI) >= {args.thr} | Scris: {args.out}")

if __name__ == "__main__":
    main()
