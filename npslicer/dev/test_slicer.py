"""
DEV-ONLY minimal planar slicer — NOT part of the print workflow
===============================================================
Stand-in for the PrusaSlicer GUI step so the pipeline can be tested
end-to-end in environments without PrusaSlicer (CI / sandboxes). It
slices a mesh into planar perimeter + zigzag-infill G-code with the
exact dialect the unwarp stage expects (relative E, ;LAYER_CHANGE / ;Z: markers,
config footer). Print quality is NOT a goal here.

  python dev/test_slicer.py --stl out/wave_warped.stl \
      --out out/wave_planar.gcode
"""

import argparse
import math
import os
import sys

import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

FIL_AREA = math.pi * (1.75 / 2.0) ** 2


def polygons_at(mesh, z):
    from shapely.ops import polygonize
    segs = trimesh.intersections.mesh_plane(
        mesh, plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if segs is None or len(segs) == 0:
        return []
    # snap endpoints to a 1 µm grid: mesh_plane emits per-triangle
    # segments whose shared endpoints differ in the last float bits,
    # which silently breaks ring closure in polygonize at some heights
    lines = [LineString([(round(s[0][0], 3), round(s[0][1], 3)),
                         (round(s[1][0], 3), round(s[1][1], 3))])
             for s in segs
             if math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) > 1e-6]
    merged = unary_union(lines)
    polys = [p for p in polygonize(merged) if p.area > 0.5]
    # nested rings: keep outer shells, subtract holes
    out = []
    for p in polys:
        hole_of = any(q is not p and q.contains(p) for q in polys)
        if not hole_of:
            holes = [q.exterior.coords for q in polys
                     if q is not p and p.contains(q)]
            out.append(Polygon(p.exterior.coords, holes))
    return out


def zigzag(poly, spacing, angle_deg):
    """Clipped parallel lines across `poly` (shapely), as coord pairs."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    diag = math.hypot(maxx - minx, maxy - miny) / 2 + spacing
    lines = []
    n = int(diag / spacing) + 1
    inner = poly.buffer(-0.4)
    if inner.is_empty:
        return lines
    for k in range(-n, n + 1):
        o = k * spacing
        p0 = (cx - diag * ca - o * sa, cy - diag * sa + o * ca)
        p1 = (cx + diag * ca - o * sa, cy + diag * sa + o * ca)
        seg = LineString([p0, p1]).intersection(inner)
        if seg.is_empty:
            continue
        geoms = getattr(seg, "geoms", [seg])
        for g in geoms:
            if isinstance(g, LineString) and g.length > 1.0:
                lines.append(list(g.coords))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer_h", type=float, default=0.2)
    ap.add_argument("--width", type=float, default=0.45)
    ap.add_argument("--infill_spacing", type=float, default=0.9)
    args = ap.parse_args()

    mesh = trimesh.load(args.stl, force="mesh")
    zmax = float(mesh.bounds[1, 2])
    g = ["; DEV test slicer (planar) — for pipeline testing only",
         "G21", "G90", "M83", "G92 E0",
         "G1 Z5 F600", "G1 X10 Y10 F9000"]
    e_rate = args.layer_h * args.width / FIL_AREA
    z = args.layer_h
    ang = 45.0
    n_layers = 0
    while z <= zmax + 1e-6:
        polys = polygons_at(mesh, min(z - args.layer_h / 2 + 1e-4,
                                      zmax - 1e-4))
        g.append(";LAYER_CHANGE")
        g.append(f";Z:{z:.2f}")
        g.append(f"G1 Z{z:.3f} F600")
        for poly in polys:
            for ring in [poly.exterior] + list(poly.interiors):
                cs = list(ring.coords)
                g.append(f"G1 X{cs[0][0]:.3f} Y{cs[0][1]:.3f} F9000")
                for (xa, ya), (xb, yb) in zip(cs[:-1], cs[1:]):
                    L = math.hypot(xb - xa, yb - ya)
                    if L < 1e-6:
                        continue
                    g.append(f"G1 X{xb:.3f} Y{yb:.3f} "
                             f"E{L * e_rate:.5f} F2400")
            for line in zigzag(poly, args.infill_spacing, ang):
                g.append(f"G1 X{line[0][0]:.3f} Y{line[0][1]:.3f} F9000")
                for (xa, ya), (xb, yb) in zip(line[:-1], line[1:]):
                    L = math.hypot(xb - xa, yb - ya)
                    g.append(f"G1 X{xb:.3f} Y{yb:.3f} "
                             f"E{L * e_rate:.5f} F3000")
        ang = 135.0 if ang == 45.0 else 45.0
        z = round(z + args.layer_h, 6)
        n_layers += 1
    g += ["G1 E-0.8 F2100", f"G1 Z{zmax + 2:.2f} F600", "M84",
          "; prusa_slicer_config = begin",
          "; use_relative_e_distances = 1",
          "; filament_diameter = 1.75",
          "; prusa_slicer_config = end"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(g))
    print(f"{n_layers} layers -> {args.out} ({len(g)} lines)")


if __name__ == "__main__":
    main()
