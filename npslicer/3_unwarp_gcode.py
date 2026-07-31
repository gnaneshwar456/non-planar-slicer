"""
NP-Slicer — stage 3: inverse-warp the planar G-code
======================================================
Input:  G-code exported from the PrusaSlicer GUI for <stem>_warped.stl
        (relative E, no arcs) + the stage-1 field.
Output: <stem>_final.gcode — the non-planar file for the MK4.

The stage-1 field carries the MODE; the two inverse maps differ:

SHEAR mode:
  * every move after the first ;LAYER_CHANGE gets z_real = z_w + S(x,y),
    sampled along the segment; E is redistributed by 2D length share
    only — a pure shear preserves layer thickness, NO thinning term
    (validated). Start G-code / purge stay untouched.
  * real z >= planar z here, so travels HOP over the sampled real
    surface (and support towers) along their path, then drop.
  * raised undersides float in real space: grid supports (bed ->
    B - gap) are generated from the field's own B and spliced in
    level-by-level, PACED with the build (a support level prints only
    once the real build height in its own neighbourhood reaches it —
    supports never tower over the print). Marked ; NP-SUPPORT-* (the
    verifier recognizes them). PrusaSlicer supports must be OFF: the
    warped part sits flat so the slicer sees nothing to support, and
    slicer supports would collide with the spliced ones — a warning
    fires if support G-code is detected in the input.

TOP mode:
  * below the blend zone the map is IDENTITY — start G-code, purge,
    first layers, most supports are untouched by construction;
  * extrusion moves inside the blend zone are subdivided, Z is mapped
    to real space, and E is redistributed per sub-segment as
        E_k = E_total * (L2d_k * th_k) / sum(L2d_j * th_j)
    where th is the local layer-thickness factor of the map. (For a
    bead printed on a slope, path length grows by sec(slope) while the
    deposited cross-section shrinks by cos(slope) — they cancel, so
    the horizontal run times the vertical thickness factor is the
    exact volume model. This extends Afshari's 3D-distance extrusion
    model with the thickness term pure shear doesn't need.)
  * travel moves are hopped OVER the real surface along their path
    (sampled max of the mapped heights), then dropped to the real
    destination height — never a straight chord through a ridge.

Safety property inherited from the map: real z <= planar z, so nothing
we emit is ever HIGHER than what PrusaSlicer planned; combined with
the monotone map, planar collision ordering survives.

Run:
  python 3_unwarp_gcode.py --field outputs/wave_field.npz \
      --gcode inputs/wave_sliced.gcode
"""

import argparse
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as vc                                    # noqa: E402

G1_RE = re.compile(r"^G([01])((?:\s+[XYZEF]-?\d*\.?\d+)+)\s*(;.*)?$")
ARC_RE = re.compile(r"^G[23](\s|$)")
Z_RE = re.compile(vc.Z_MARK_RE)


def parse_fields(body):
    out = {}
    for m in re.finditer(r"([XYZEF])(-?\d*\.?\d+)", body):
        out[m.group(1)] = float(m.group(2))
    return out


def parse_config(lines):
    """PrusaSlicer dumps '; key = value' config lines in the footer."""
    cfg = {}
    for ln in lines:
        m = re.match(r"^; ([a-z_0-9]+) = (.*)$", ln)
        if m:
            cfg[m.group(1)] = m.group(2)
    return cfg


def derived_config(lines):
    """Numeric print settings the support generator needs, from the
    PrusaSlicer config dump (Prusa spelling only)."""
    cfg = parse_config(lines)

    def f(key, d):
        v = cfg.get(key)
        if v is None:
            return d
        try:
            return float(v.split(",")[0])
        except ValueError:
            return d

    return {
        "retract_length": f("retract_length", 0.8),
        "retract_speed": f("retract_speed", 35.0),
        "travel_speed": f("travel_speed", 130.0),
        "filament_diameter": f("filament_diameter", 1.75),
        "layer_height": f("layer_height", 0.2),
    }


def raster_runs(mask, xs, ys, spacing, along_x, inset=0.35,
                min_len=1.0):
    """Straight raster segments across True runs of `mask`, rows/cols
    `spacing` apart — numpy-only stand-in for a shapely infill
    (identical output for the axis-aligned angles used). Serpentine
    ordering keeps hops short."""
    segs = []
    if along_x:
        step = max(1, int(round(spacing / (ys[1] - ys[0]))))
        for i in range(0, mask.shape[0], step):
            row = np.flatnonzero(mask[i])
            if row.size == 0:
                continue
            brk = np.where(np.diff(row) > 1)[0]
            for run in np.split(row, brk + 1):
                x0, x1 = xs[run[0]] + inset, xs[run[-1]] - inset
                if x1 - x0 >= min_len:
                    segs.append([x0, ys[i], x1, ys[i]])
    else:
        step = max(1, int(round(spacing / (xs[1] - xs[0]))))
        for j in range(0, mask.shape[1], step):
            col = np.flatnonzero(mask[:, j])
            if col.size == 0:
                continue
            brk = np.where(np.diff(col) > 1)[0]
            for run in np.split(col, brk + 1):
                y0, y1 = ys[run[0]] + inset, ys[run[-1]] - inset
                if y1 - y0 >= min_len:
                    segs.append([xs[j], y0, xs[j], y1])
    for k in range(1, len(segs), 2):          # serpentine
        s = segs[k]
        segs[k] = [s[2], s[3], s[0], s[1]]
    return segs


def support_levels(B, sup_mask, xs, ys, layer_h, cfg, fil_area,
                   speed=40.0, spacing=2.0, bead_w=0.5):
    """Grid supports bed -> underside-gap, relative E, self-contained.
    Support-block generator (validated); returns
    [(z_level, mask_at_level, gcode_lines)], top_of_supports."""
    if not sup_mask.any():
        return [], 0.0
    gap = layer_h
    top = np.where(sup_mask, B - gap, -np.inf)
    z_top_max = float(top[sup_mask].max())
    R = cfg["retract_length"]
    rf = cfg["retract_speed"] * 60
    tf = cfg["travel_speed"] * 60
    pf = speed * 60
    levels = []          # spliced level-by-level, paced with the build
    z = 0.3
    along_x = True
    while z <= z_top_max + 1e-9:
        mask_z = sup_mask & (z <= top)
        if mask_z.any():
            segs = raster_runs(mask_z, xs, ys, spacing, along_x)
            if segs:
                g = [f"; NP-SUPPORT-LEVEL z={z:.2f}", "M83",
                     f"G1 E-{R:.5f} F{rf:.0f}",
                     f"G1 Z{z + 1.0:.3f} F600"]
                first = True
                for (x0, y0, x1, y1) in segs:
                    if not first:
                        g.append(f"G1 E-{R:.5f} F{rf:.0f}")
                        g.append(f"G1 Z{z + 1.0:.3f} F600")
                    g.append(f"G1 X{x0:.3f} Y{y0:.3f} F{tf:.0f}")
                    g.append(f"G1 Z{z:.3f} F600")
                    g.append(f"G1 E{R:.5f} F{rf:.0f}")
                    L = math.hypot(x1 - x0, y1 - y0)
                    de = L * layer_h * bead_w / fil_area
                    g.append(f"G1 X{x1:.3f} Y{y1:.3f} E{de:.5f} "
                             f"F{pf:.0f}")
                    first = False
                g.append(f"G1 E-{R:.5f} F{rf:.0f}")
                g.append(f"G1 Z{z + 1.0:.3f} F600")
                g.append(f"G1 E{R:.5f} F{rf:.0f}")
                g.append("; NP-SUPPORT-LEVEL-END")
                lx, ly = segs[-1][2], segs[-1][3]
                levels.append((z, mask_z, g, (lx, ly)))
        along_x = not along_x
        z = round(z + layer_h, 5)
    return levels, z_top_max


def unwarp_shear(wf, lines, out, args):
    """inverse shear + support splicing on the field."""
    xs, ys = wf.xs, wf.ys
    S = wf.S
    B = np.asarray(wf.raw["B"], float)
    B = np.where(np.isfinite(B), B, 0.0)
    sup_mask = np.asarray(wf.raw["sup_mask"], bool)
    cfg = derived_config(lines)
    layer_h = cfg["layer_height"]
    fil_area = math.pi * (cfg["filament_diameter"] / 2.0) ** 2

    # PrusaSlicer supports must be OFF in shear mode: the warped part
    # sits flat (nothing for the slicer to support) and slicer supports
    # would be sheared into the spliced NP support region.
    n_sup_gc = sum(1 for ln in lines
                   if ln.startswith(";TYPE:Support"))
    if n_sup_gc:
        print(f"WARNING: the sliced G-code contains PrusaSlicer "
              f"support material ({n_sup_gc} ;TYPE:Support markers). "
              f"In SHEAR mode slice with supports OFF — stage 3 "
              f"splices its own NP supports. Re-slice and re-upload; "
              f"continuing anyway, but expect verifier failures.")

    if getattr(args, "supports", "auto") == "off" and sup_mask.any():
        print("WARNING: --supports off — skipping NP supports. The "
              "raised underside will print UNSUPPORTED (bridging/sag "
              "risk); only safe where the spans are short enough to "
              "bridge. Inspect the underside.")
        sup_mask = np.zeros_like(sup_mask)
    raw_levels, sup_top = support_levels(B, sup_mask, xs, ys, layer_h,
                                         cfg, fil_area)
    # pacing: a support level may print only once the REAL build height
    # in its own neighbourhood has reached it — supports grow in sync
    # with the print, never towering over it. The part arrives at a
    # floating cell at real B(c); level completion by then is
    # guaranteed because S_local >= S(c). (validated.)
    sup_levels = [(z, float(S[mask].max()) if mask.any() else 0.0, g,
                   last_xy)
                  for (z, mask, g, last_xy) in raw_levels]
    print(f"Support levels: {len(sup_levels)} "
          f"(tops up to {sup_top:.2f} mm)")

    S_max = wf.S_max
    # obstacle field for travel planning: support towers are absolute-Z
    # obstacles; the print surface is layer-relative (zp + S)
    tower = np.where(sup_mask, B - 0.2, 0.0)
    hf_tw = vc.HeightField(xs, ys, tower)
    has_tower = bool(sup_mask.any())

    def warp_z(px, py, pz):
        return float(wf.z_real(px, py, pz))

    # (rx, ry) = the nozzle's REAL last XY — equal to the tracked input
    # position except right after a spliced support block
    def travel_height(px0, py0, px1, py1, pz):
        L = math.hypot(px1 - px0, py1 - py0)
        n = max(2, int(L / 2.0) + 1)
        ts = np.linspace(0.0, 1.0, n)
        sxs = px0 + ts * (px1 - px0)
        sys_ = py0 + ts * (py1 - py0)
        zt = pz + float(np.max(wf.hf_s.z(sxs, sys_)))
        if has_tower:
            zt = max(zt, float(np.max(hf_tw.z(sxs, sys_))))
        return zt + 0.4

    layer_mark = getattr(args, "layer_mark", vc.LAYER_MARK)
    z_re = re.compile(getattr(args, "z_mark_re", vc.Z_MARK_RE))
    body_end_str = vc.GCODE_FLAVORS.get(
        "bambu" if layer_mark == "; CHANGE_LAYER" else "prusa")["body_end"]
    first_layer_idx = next(i for i, ln in enumerate(lines)
                           if ln.startswith(layer_mark))
    # machine end gcode (park / cooldown) is kept VERBATIM — never warped
    body_end_idx = next((i for i, ln in enumerate(lines)
                         if ln.startswith(body_end_str)), len(lines))
    header = [f"; NP-Slicer non-planar G-code (SHEAR mode)",
              f"; field: {os.path.basename(args.field)}  "
              f"S_max={S_max:.2f}  "
              f"theta={wf.meta['theta_deg']:.1f}deg  "
              f"supports={len(sup_levels)} NP levels",
              f"; do not edit — regenerate with 3_unwarp_gcode.py"]
    out_lines = header + lines[:first_layer_idx]

    # seed the position tracker from the start G-code (purge line etc.)
    x = y = zp = None
    for ln0 in lines[:first_layer_idx]:
        m0 = G1_RE.match(ln0.strip())
        if m0:
            f0 = parse_fields(m0.group(2))
            x = f0.get("X", x)
            y = f0.get("Y", y)
            zp = f0.get("Z", zp)

    sup_i = 0
    cur_layer_zp = 0.0
    n_warped = n_travel = n_pass = 0
    e_in = e_out = e_sup = 0.0
    rx, ry = x, y                     # real nozzle XY (see above)
    for (z, _sl, g, _lxy) in sup_levels:
        for gl in g:
            m = re.search(r"E(-?\d*\.?\d+)", gl)
            if m:
                e_sup += float(m.group(1))

    for ln in lines[first_layer_idx:body_end_idx]:
        zm = z_re.match(ln.strip())
        if zm:
            try:
                cur_layer_zp = float(zm.group(1))
                zp = cur_layer_zp
            except ValueError:
                pass
            # splice support levels the moment the build's real top
            # reaches them (supports never tower over the print)
            spliced = False
            while sup_i < len(sup_levels):
                z_lvl, s_loc, block, last_xy = sup_levels[sup_i]
                if z_lvl > cur_layer_zp + s_loc + 0.01:
                    break
                if not spliced:
                    out_lines.append(ln)   # keep ;Z: before the block
                    spliced = True
                out_lines.extend(block)
                rx, ry = last_xy           # real nozzle moved here
                sup_i += 1
            if spliced:
                continue
        m = G1_RE.match(ln.strip())
        if not m:
            am = ARC_RE.match(ln.strip())
            if am:
                fa = parse_fields(ln)
                # Bambu spiral layer-change lift: Z-only arc, same XY.
                # Map its Z into real space (z_w + S) and emit a plain
                # G1 Z — the anti-seam spiral is cosmetic on a travel.
                if "X" not in fa and "Y" not in fa and "Z" in fa:
                    px = x if x is not None else rx
                    py = y if y is not None else ry
                    if px is not None and py is not None:
                        zr = warp_z(px, py, fa["Z"])
                        fp = f" F{fa['F']:.0f}" if "F" in fa else ""
                        out_lines.append(f"G1 Z{zr:.3f}{fp} ; np: Bambu "
                                         f"spiral->linear Z")
                        zp = fa["Z"]
                        continue
            out_lines.append(ln)
            continue
        fl = parse_fields(m.group(2))
        nx = fl.get("X", x)
        ny = fl.get("Y", y)
        nzp = fl.get("Z", zp)
        de = fl.get("E")
        ff = fl.get("F")
        moved = ("X" in fl or "Y" in fl or "Z" in fl)
        if de is not None:
            e_in += de
        if not moved or nx is None or ny is None or nzp is None \
                or x is None or y is None or zp is None:
            # E/F-only lines pass through; positioning moves with an
            # incomplete position history are rewritten conservatively
            # (shear-mode rule): never send a raw planar Z — in shear mode the
            # real surface is ABOVE planar, so park high instead.
            if not moved:
                out_lines.append(ln)
                if de is not None:
                    e_out += de
            elif nx is not None and ny is not None and nzp is not None:
                zr = warp_z(nx, ny, nzp)
                f_part = f" F{ff:.0f}" if ff is not None else ""
                e_part = f" E{de:.5f}" if de is not None else ""
                if de is not None:
                    e_out += de
                out_lines.append(f"G1 X{nx:.3f} Y{ny:.3f}"
                                 f"{e_part}{f_part}")
                out_lines.append(f"G1 Z{zr:.3f} F600")
                rx, ry = nx, ny
            elif nzp is not None:
                z_park = nzp + S_max + 0.6
                f_part = f" F{ff:.0f}" if ff is not None else ""
                out_lines.append(f"G1 Z{z_park:.3f}{f_part} ; parked "
                                 f"(XY unknown yet)")
                if de is not None:
                    out_lines.append(f"G1 E{de:.5f}")
                    e_out += de
            else:
                out_lines.append(ln)
                if de is not None:
                    e_out += de
            x, y, zp = nx, ny, nzp
            n_pass += 1
            continue
        # travel: hop over the real obstacles along the path, then
        # DROP to the real height at the destination
        if de is None or de <= 0:
            hx = rx if rx is not None else x
            hy = ry if ry is not None else y
            zt = travel_height(hx, hy, nx, ny, nzp)
            zdst = warp_z(nx, ny, nzp)
            f_part = f" F{ff:.0f}" if ff is not None else ""
            if de is not None:            # retract distributed first
                out_lines.append(f"G1 E{de:.5f}{f_part}")
                e_out += de
                f_part = ""
            out_lines.append(f"G1 Z{zt:.3f} F600")
            out_lines.append(f"G1 X{nx:.3f} Y{ny:.3f}{f_part}")
            out_lines.append(f"G1 Z{zdst:.3f} F600")
            x, y, zp = nx, ny, nzp
            rx, ry = nx, ny
            n_travel += 1
            continue
        # extrusion: subdivide, shear Z, redistribute E by 2D length
        # share ONLY — pure shear preserves layer thickness, so no
        # thinning term (validated extrusion model)
        L = math.hypot(nx - x, ny - y)
        n = max(1, int(math.ceil(L / args.seg)))
        ts = np.linspace(0.0, 1.0, n + 1)
        sxs = x + ts * (nx - x)
        sys_ = y + ts * (ny - y)
        szs = zp + ts * (nzp - zp)
        zreal = wf.z_real(sxs, sys_, szs)
        L2 = np.hypot(np.diff(sxs), np.diff(sys_))
        L2sum = float(L2.sum())
        emitted_f = False
        for k in range(n):
            ek = (de * (L2[k] / L2sum) if L2sum > 1e-12 else de / n)
            seg = [f"G1 X{sxs[k+1]:.3f} Y{sys_[k+1]:.3f} "
                   f"Z{zreal[k+1]:.3f} E{ek:.5f}"]
            if ff is not None and not emitted_f:
                seg.append(f"F{ff:.0f}")
                emitted_f = True
            out_lines.append(" ".join(seg))
            e_out += ek
        n_warped += 1
        x, y, zp = nx, ny, nzp
        rx, ry = nx, ny

    while sup_i < len(sup_levels):        # flush any remaining levels
        out_lines.extend(sup_levels[sup_i][2])
        sup_i += 1

    out_lines += lines[body_end_idx:]     # machine end gcode, verbatim

    with open(out, "w") as f:
        f.write("\n".join(out_lines))
    print(f"Unwarped {n_warped} extrusion moves, {n_travel} travels "
          f"({n_pass} passed through) -> {out}")
    print(f"Extrusion: in {e_in:.1f} mm, out {e_out:.1f} mm "
          f"({(e_out/e_in*100 if e_in else 100):.1f}% — pure shear, "
          f"no thinning) + {e_sup:.1f} mm NP supports")
    print(f'Next: python 4_verify_gcode.py --gcode "{out}" '
          f'--field "{args.field}"')


def main():
    ap = argparse.ArgumentParser(description="stage 3: unwarp G-code")
    ap.add_argument("--field", required=True)
    ap.add_argument("--gcode", required=True,
                    help="Planar G-code sliced from <stem>_warped.stl")
    ap.add_argument("--out", default=None,
                    help="Default: <field dir>/<stem>_final.gcode")
    ap.add_argument("--seg", type=float, default=0.8,
                    help="Max sub-segment length inside the blend zone "
                         "(mm)")
    ap.add_argument("--supports", choices=["auto", "on", "off"],
                    default="auto",
                    help="Shear-mode NP supports under the raised "
                         "underside: auto/on splice them (default, current "
                         "behaviour); off skips them entirely, so the "
                         "underside prints UNSUPPORTED (bridging/sag risk "
                         "— only safe for bridgeable spans). No effect in "
                         "top mode (PrusaSlicer handles those supports).")
    args = ap.parse_args()

    wf = vc.load_field(args.field)
    stem = os.path.splitext(os.path.basename(args.field))[0]
    stem = stem[:-6] if stem.endswith("_field") else stem
    outdir = os.path.dirname(os.path.abspath(args.field))
    out = args.out or os.path.join(outdir, f"{stem}_final.gcode")

    if not os.path.exists(args.gcode):
        raise SystemExit(f"ERROR: G-code not found: {args.gcode!r}")
    lines = open(args.gcode, errors="replace").read().splitlines()

    # --- G-code flavor (PrusaSlicer default; Bambu Studio / OrcaSlicer
    # for the P1S) — pick the layer/Z markers and body delimiter ---------
    flavor, mk = vc.detect_gcode_flavor(lines)
    args.layer_mark = mk["layer_mark"]
    args.z_mark_re = mk["z_mark_re"]
    print(f"G-code flavor: {flavor} (layer '{mk['layer_mark']}', "
          f"Z '{mk['z_mark_re']}')")

    # --- input sanity: the failure modes fail loudly, not silently ----
    # scan only the MODEL BODY (first layer -> machine end); start/end
    # machine macros legitimately contain arcs (nozzle wipe circles) that
    # are not part of the print.
    body_start = next((i for i, ln in enumerate(lines)
                       if ln.startswith(mk["layer_mark"])), 0)
    body_end = next((i for i, ln in enumerate(lines)
                     if ln.startswith(mk["body_end"])), len(lines))
    for i in range(body_start, body_end):
        s = lines[i].strip()
        if ARC_RE.match(s):
            fa = parse_fields(s)
            if "X" in fa or "Y" in fa:
                raise SystemExit(
                    f"ERROR: planar arc (G2/G3) at line {i+1}. Disable "
                    f"'Arc fitting' in Print/Quality settings and "
                    f"re-export.")
            # Z-only arc = Bambu's spiral layer-change lift; converted to
            # a mapped linear Z during unwarp (not a model arc).
    cfg = parse_config(lines)
    if cfg and cfg.get("use_relative_e_distances", "0") not in ("1", "true"):
        raise SystemExit("ERROR: G-code uses ABSOLUTE E. Enable 'Use "
                         "relative E distances' (PrusaSlicer: Printer "
                         "Settings; Bambu/Orca: usually already relative) "
                         "and re-export.")
    if not any(ln.startswith(mk["layer_mark"]) for ln in lines):
        raise SystemExit(f"ERROR: no '{mk['layer_mark']}' layer markers "
                         f"found — unrecognized slicer output?")

    # --- part-position sanity: catch the BED-SIZE / auto-center trap ----
    # A field is tied to WHERE the part sits on the plate (stage 1 centers
    # the part on the profile's bed). If the warped STL is sliced for a
    # DIFFERENT bed, the slicer re-centers the object and the shear/top
    # field is then sampled at the wrong XY -> distorted print. Fail loud.
    fx0, fx1 = float(wf.xs.min()), float(wf.xs.max())
    fy0, fy1 = float(wf.ys.min()), float(wf.ys.max())
    _px, _py, _x, _y = [], [], None, None
    for ln in lines[body_start:body_end]:
        s = ln.strip()
        if s[:2] in ("G1", "G0"):
            fld = parse_fields(s)
            if "X" in fld:
                _x = fld["X"]
            if "Y" in fld:
                _y = fld["Y"]
            if fld.get("E", 0.0) > 0 and _x is not None and _y is not None:
                _px.append(_x)
                _py.append(_y)
    if _px:
        apx, apy = np.array(_px), np.array(_py)
        inside = float(((apx >= fx0) & (apx <= fx1) &
                        (apy >= fy0) & (apy <= fy1)).mean())
        if inside < 0.90:
            cgx = 0.5 * (np.percentile(apx, 2) + np.percentile(apx, 98))
            cgy = 0.5 * (np.percentile(apy, 2) + np.percentile(apy, 98))
            raise SystemExit(
                f"ERROR: the sliced part does not line up with the warp "
                f"field — only {inside*100:.0f}% of extrusion falls inside "
                f"the field grid. Sliced part center ~({cgx:.0f},{cgy:.0f}) "
                f"vs field center ~({0.5*(fx0+fx1):.0f},"
                f"{0.5*(fy0+fy1):.0f}). This is almost always a BED-SIZE / "
                f"auto-center mismatch: the field was built for bed "
                f"{wf.meta.get('bed_x')}x{wf.meta.get('bed_y')} but the "
                f"G-code was sliced for a different bed and the slicer "
                f"re-centered the object. FIX: rebuild the field with the "
                f"SAME --printer you slice for (e.g. --printer bambu_p1s), "
                f"re-warp, and slice the new warped STL WITHOUT moving it "
                f"on the plate.")

    if wf.meta.get("passthrough"):
        with open(out, "w") as f:
            f.write("\n".join(
                ["; NP-Slicer — PLANAR PASSTHROUGH (no warp "
                 "applied — flat geometry or near-zero benefit)"]
                + lines))
        print(f"PLANAR PASSTHROUGH -> {out} (G-code unchanged)")
        return

    if args.supports == "off" and wf.mode != "shear":
        print("NOTE: --supports off has no effect in TOP mode "
              "(PrusaSlicer generates those supports).")

    if wf.mode == "shear":
        return unwarp_shear(wf, lines, out, args)

    z_id = wf.H - wf.d_blend          # identity below this warped height
    layer_mark = getattr(args, "layer_mark", vc.LAYER_MARK)
    body_end_str = vc.GCODE_FLAVORS.get(
        "bambu" if layer_mark == "; CHANGE_LAYER" else "prusa")["body_end"]
    body_end_idx = next((i for i, ln in enumerate(lines)
                         if ln.startswith(body_end_str)), len(lines))
    n_warped = n_travel = n_pass = 0
    e_in = e_out = 0.0
    th_min = 1.0
    out_lines = [f"; NP-Slicer non-planar G-code",
                 f"; field: {os.path.basename(args.field)}  "
                 f"H={wf.H:.2f}  d_blend={wf.d_blend:.2f}  "
                 f"theta={wf.meta['theta_deg']:.1f}deg",
                 f"; do not edit — regenerate with 3_unwarp_gcode.py"]

    x = y = zp = None                 # tracked PLANAR position

    def zr(px, py, pz):
        return float(wf.z_real(px, py, pz))

    for ln in lines[:body_end_idx]:
        s = ln.strip()
        m = G1_RE.match(s)
        if not m:
            am = ARC_RE.match(s)
            if am:
                fa = parse_fields(s)
                # Bambu spiral layer-change lift: Z-only arc, same XY ->
                # map Z to real space, emit a plain G1 Z.
                if ("X" not in fa and "Y" not in fa and "Z" in fa
                        and x is not None and y is not None):
                    zrr = zr(x, y, fa["Z"])
                    fp = f" F{fa['F']:.0f}" if "F" in fa else ""
                    out_lines.append(f"G1 Z{zrr:.3f}{fp} ; np: Bambu "
                                     f"spiral->linear Z")
                    zp = fa["Z"]
                    continue
            out_lines.append(ln)
            continue
        fl = parse_fields(m.group(2))
        nx = fl.get("X", x)
        ny = fl.get("Y", y)
        nzp = fl.get("Z", zp)
        de = fl.get("E")
        ff = fl.get("F")
        moved = ("X" in fl or "Y" in fl or "Z" in fl)
        if de is not None:
            e_in += de

        # -- no motion, or position not yet known: pass through ---------
        if not moved or nzp is None:
            out_lines.append(ln)
            x, y, zp = nx, ny, nzp
            n_pass += 1
            continue
        # below the blend zone the map is identity for BOTH endpoints ->
        # emit verbatim (keeps start G-code, purge, first layers,
        # near-bed supports byte-identical)
        if (zp is None or zp <= z_id) and nzp <= z_id:
            out_lines.append(ln)
            if de is not None:
                e_out += de
            x, y, zp = nx, ny, nzp
            n_pass += 1
            continue
        # XY unknown but Z in the blend zone (shouldn't happen in
        # PrusaSlicer output after the purge line): park at planar Z,
        # which is always >= real Z -> safe.
        if nx is None or ny is None:
            out_lines.append(ln + " ; planar Z kept (XY unknown, "
                                  "planar >= real is safe)")
            x, y, zp = nx, ny, nzp
            n_pass += 1
            continue

        f_part = f" F{ff:.0f}" if ff is not None else ""

        # -- travel (no positive extrusion) ------------------------------
        if de is None or de <= 0:
            e_part = f" E{de:.5f}" if de is not None else ""
            if x is None or y is None:
                out_lines.append(f"G1 X{nx:.3f} Y{ny:.3f} "
                                 f"Z{nzp:.3f}{e_part}{f_part}")
            else:
                L = math.hypot(nx - x, ny - y)
                n = max(2, int(L / 2.0) + 1)
                ts = np.linspace(0.0, 1.0, n)
                sxs = x + ts * (nx - x)
                sys_ = y + ts * (ny - y)
                szs = (zp if zp is not None else nzp) + \
                    ts * (nzp - (zp if zp is not None else nzp))
                z_need = float(np.max(wf.z_real(sxs, sys_,
                                                np.maximum(szs, nzp))))
                z_dst = zr(nx, ny, nzp)
                z_hop = min(max(z_need + 0.4, z_dst), nzp + 0.6)
                if de is not None:            # retract distributed first
                    out_lines.append(f"G1 E{de:.5f}{f_part}")
                    e_out += de
                    f_part = ""
                out_lines.append(f"G1 Z{z_hop:.3f} F600")
                out_lines.append(f"G1 X{nx:.3f} Y{ny:.3f}{f_part}")
                out_lines.append(f"G1 Z{z_dst:.3f} F600")
            x, y, zp = nx, ny, nzp
            n_travel += 1
            continue

        # -- extrusion move: subdivide, map Z, redistribute E ------------
        if x is None or y is None or zp is None:
            out_lines.append(ln)      # cannot map without a start point
            e_out += de
            x, y, zp = nx, ny, nzp
            n_pass += 1
            continue
        L = math.hypot(nx - x, ny - y)
        n = max(1, int(math.ceil(L / args.seg)))
        ts = np.linspace(0.0, 1.0, n + 1)
        sxs = x + ts * (nx - x)
        sys_ = y + ts * (ny - y)
        szs = zp + ts * (nzp - zp)
        zreal = wf.z_real(sxs, sys_, szs)
        mx = 0.5 * (sxs[1:] + sxs[:-1])
        my = 0.5 * (sys_[1:] + sys_[:-1])
        mz = 0.5 * (szs[1:] + szs[:-1])
        th = np.asarray(wf.thickness(mx, my, mz), float)
        th_min = min(th_min, float(th.min()) if th.size else 1.0)
        L2 = np.hypot(np.diff(sxs), np.diff(sys_))
        L2sum = float(L2.sum())
        emitted_f = False
        for k in range(n):
            # length share of the planar E, SCALED by the local layer
            # thinning (not normalized — total E must shrink where the
            # map compresses layers, or the blend zone over-extrudes)
            ek = (de * (L2[k] / L2sum) * float(th[k])
                  if L2sum > 1e-12 else de / n)
            seg = [f"G1 X{sxs[k+1]:.3f} Y{sys_[k+1]:.3f} "
                   f"Z{zreal[k+1]:.3f} E{ek:.5f}"]
            if ff is not None and not emitted_f:
                seg.append(f"F{ff:.0f}")
                emitted_f = True
            out_lines.append(" ".join(seg))
            e_out += ek
        n_warped += 1
        x, y, zp = nx, ny, nzp

    out_lines += lines[body_end_idx:]     # machine end gcode, verbatim

    with open(out, "w") as f:
        f.write("\n".join(out_lines))
    print(f"Unwarped {n_warped} extrusion moves, {n_travel} travels "
          f"({n_pass} passed through) -> {out}")
    print(f"Extrusion: in {e_in:.1f} mm, out {e_out:.1f} mm "
          f"({(e_out/e_in*100 if e_in else 100):.1f}% — thinning "
          f"compensation; min thickness factor {th_min:.2f})")
    print(f'Next: python 4_verify_gcode.py --gcode "{out}" '
          f'--field "{args.field}"')


if __name__ == "__main__":
    main()
