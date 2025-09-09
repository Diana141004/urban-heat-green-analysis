import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--cmap", default="inferno")
ap.add_argument("--label", default="")
ap.add_argument("--pclip", type=float, default=2)
ap.add_argument("--sym", action="store_true")
ap.add_argument("--vmin", type=float, default=None)
ap.add_argument("--vmax", type=float, default=None)
# NOI:
ap.add_argument("--width",  type=int, default=None)   # lățime PNG în pixeli
ap.add_argument("--height", type=int, default=None)   # în pixeli (se păstrează aspect ratio dacă lipsește unul)
ap.add_argument("--scale",  type=float, default=None) # factor vs raster (1.0 = 1:1, 0.5 = jumătate)
ap.add_argument("--interp", default="nearest", choices=["nearest","none","bilinear","bicubic"])
ap.add_argument("--dpi", type=int, default=300)       # dpi intern pt. Matplotlib
ap.add_argument("--aspect", default="auto", choices=["auto","equal"])
args = ap.parse_args()

with rasterio.open(args.input) as src:
    a = src.read(1).astype("float32"); nodata = src.nodata
    H, W = a.shape

if nodata is not None:
    a = np.where(a == nodata, np.nan, a)

finite = np.isfinite(a)
if not finite.any():
    raise SystemExit("No finite pixels!")

# vmin/vmax
if args.vmin is not None and args.vmax is not None:
    vmin, vmax = args.vmin, args.vmax
else:
    p_lo, p_hi = np.nanpercentile(a[finite], (args.pclip, 100-args.pclip))
    if args.sym:
        m = max(abs(p_lo), abs(p_hi)); vmin, vmax = -m, m
    else:
        vmin, vmax = p_lo, p_hi

# dimensiuni PNG țintă
if args.scale is not None:
    target_w = int(W * args.scale); target_h = int(H * args.scale)
elif args.width or args.height:
    if args.width and args.height:
        target_w, target_h = args.width, args.height
    elif args.width:
        target_w = args.width; target_h = int(H * (args.width / W))
    else:
        target_h = args.height; target_w = int(W * (args.height / H))
else:
    target_w, target_h = 1600, 1200  # fallback

fig_w_in = target_w / args.dpi
fig_h_in = target_h / args.dpi

plt.figure(figsize=(fig_w_in, fig_h_in), dpi=args.dpi)
im = plt.imshow(a, vmin=vmin, vmax=vmax, cmap=args.cmap,
                interpolation=args.interp, aspect=args.aspect)
cb = plt.colorbar(im)
if args.label: cb.set_label(args.label)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0)
plt.close()
