"""
NP-Slicer — stage 4: verify the final G-code before it touches the MK4
=========================================================================
Self-contained (rebuilds its own height map from the G-code itself) and
profile-driven, so a wrong hand-entered profile value fails HERE, not
on the printer. Temporal simulation in print order, four checks:

  [floor]   nothing extrudes below the bed (z < -0.05);
  [burial]  no move dives deeper below already-printed material in its
            own column than the layer-slope allowance permits;
  [cone]    Ahlers 'large angle, small distance': printed material may
            not intrude into the nozzle cone; plus the far-field
            housing floor ('small angle, large distance'): nothing
            taller than head_clearance_h within head_radius laterally;
  [lateral] flat-tip gouge: the tip land must not plow a laterally
            adjacent bead sitting higher on a sloped layer.

Exit code 0 + 'VERIFICATION PASSED' means: send it to the printer.

Run:
  python 4_verify_gcode.py --gcode outputs/wave_final.gcode \
      --field outputs/wave_field.npz
"""

import argparse
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as vc                                    # noqa: E402

G1 = re.compile(r"^G[01](?:\s+X(-?\d*\.?\d+))?(?:\s+Y(-?\d*\.?\d+))?"
                r"(?:\s+Z(-?\d*\.?\d+))?(?:\s+E(-?\d*\.?\d+))?")


def iter_moves(path, bed_x=400.0, bed_y=400.0,
               layer_mark=vc.LAYER_MARK,
               body_end="; prusa_slicer_config = begin"):
    """Yield (line_no, x0,y0,z0, x1,y1,z1, extruding) in print order,
    starting after the first layer marker, stopping at the config dump /
    machine end. Layer marker + body-end are flavor-specific (Prusa vs
    Bambu Studio)."""
    x = y = z = None
    started = False
    line_no = 0
    for raw in open(path, errors="replace"):
        line_no += 1
        ln = raw.strip()
        if ln.startswith(layer_mark):
            started = True
        if ln.startswith(body_end):
            break
        if not (ln.startswith("G1") or ln.startswith("G0")):
            continue
        m = G1.match(ln)
        if not m:
            continue
        gx, gy, gz, ge = m.groups()
        nx = float(gx) if gx else x
        ny = float(gy) if gy else y
        nz = float(gz) if gz else z
        de = float(ge) if ge else 0.0
        extr = de > 0 and (gx or gy or gz)
        if (started and None not in (x, y, z, nx, ny, nz)
                and 0 <= nx <= bed_x and 0 <= ny <= bed_y):
            yield line_no, x, y, z, nx, ny, nz, extr
        x, y, z = nx, ny, nz


def main():
    ap = argparse.ArgumentParser(description="stage 4: verify")
    ap.add_argument("--gcode", required=True)
    ap.add_argument("--field", default=None,
                    help="Stage-1 npz; supplies theta and grid. Without "
                         "it, profile values are used")
    ap.add_argument("--printer", default="prusa_mk4")
    ap.add_argument("--layer_h", type=float, default=0.2)
    ap.add_argument("--cell", type=float, default=0.30,
                    help="Height-map cell size (mm)")
    args = ap.parse_args()

    prof = vc.load_printer_profile(args.printer)
    theta = None
    slope_override = False
    if args.field and os.path.exists(args.field):
        wf = vc.load_field(args.field)
        theta = float(wf.meta["theta_deg"])
        slope_override = bool(wf.meta.get("slope_override", False))
        cone_deg = float(wf.meta["cone_deg"])
        tip_r = float(wf.meta["tip_land_radius"])
        gouge_tol = float(wf.meta["gouge_tol"])
        line_w = float(wf.meta["line_width"])
        head_h = float(wf.meta["head_clearance_h"])
        head_r = float(wf.meta["head_radius"])
        bed_x, bed_y = wf.meta["bed_x"], wf.meta["bed_y"]
    else:
        cone_deg = float(prof["nozzle_cone_angle_deg"])
        tip_r = float(prof["tip_land_radius"])
        gouge_tol = float(prof["gouge_tol_mm"])
        line_w = float(prof["nozzle_diameter"]) * 1.125
        head_h = float(prof["head_clearance_h"])
        head_r = float(prof["head_radius"])
        bed_x, bed_y = float(prof["bed_x"]), float(prof["bed_y"])
        lat, _ = vc.lateral_slope_cap_deg(tip_r, line_w, cone_deg,
                                          gouge_tol)
        theta = min(float(prof.get("theta_target_default", 20.0)),
                    cone_deg, lat)
    print(f"Verifying {args.gcode}\n  theta {theta:.1f} deg  cone "
          f"{cone_deg:.1f} deg  tip_r {tip_r:.2f}  head "
          f"{head_h:.1f}mm/{head_r:.0f}mm")
    _flavor, _mk = vc.detect_gcode_flavor(
        open(args.gcode, errors="replace").read().splitlines())
    layer_mark, body_end_mark = _mk["layer_mark"], _mk["body_end"]
    print(f"  G-code flavor: {_flavor}")

    # --- pass 0: bounds + grid ------------------------------------------
    xs_all, ys_all = [], []
    n_ext = 0
    z_floor_bad = 0
    z_floor_worst = 0.0
    for (_ln, x0, y0, z0, x1, y1, z1, extr) in iter_moves(
            args.gcode, bed_x, bed_y, layer_mark, body_end_mark):
        xs_all += [x0, x1]
        ys_all += [y0, y1]
        if extr:
            n_ext += 1
            if min(z0, z1) < -0.05:
                z_floor_bad += 1
                z_floor_worst = min(z_floor_worst, min(z0, z1))
    if not xs_all:
        raise SystemExit("ERROR: no moves parsed — wrong file?")
    print(f"Parsed: {n_ext} extrusion moves")
    ok_floor = z_floor_bad == 0
    print(f"[floor]   {z_floor_bad} extrusions below bed "
          f"(worst {z_floor_worst:.3f} mm)  "
          f"{'PASS' if ok_floor else 'FAIL'}")

    cell = args.cell
    x0g, x1g = min(xs_all) - 1, max(xs_all) + 1
    y0g, y1g = min(ys_all) - 1, max(ys_all) + 1
    nx = int((x1g - x0g) / cell) + 3
    ny = int((y1g - y0g) / cell) + 3
    xs = x0g + np.arange(nx) * cell
    ys = y0g + np.arange(ny) * cell
    XX, YY = np.meshgrid(xs, ys)

    # --- pass 1: burial + cone + housing (temporal) ----------------------
    M = np.full((ny, nx), -np.inf)
    zmax = -np.inf
    cell_diag = math.hypot(cell, cell)
    allow = cell_diag * math.tan(math.radians(theta)) + 2.0 * args.layer_h
    tan_t = math.tan(math.radians(min(cone_deg + 5.0, 85.0)))
    Bc = allow + cell_diag
    n_bury = n_cone = 0
    w_bury = w_cone = 0.0
    bury_loc = cone_loc = None
    for (ln_no, x0, y0, z0, x1, y1, z1, extr) in iter_moves(
            args.gcode, bed_x, bed_y, layer_mark, body_end_mark):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(L / 0.5) + 1)
        ts = np.linspace(0, 1, n)
        sx = x0 + ts * (x1 - x0)
        sy = y0 + ts * (y1 - y0)
        sz = z0 + ts * (z1 - z0)
        jj = np.clip(np.round((sx - xs[0]) / cell).astype(int), 0, nx - 1)
        ii = np.clip(np.round((sy - ys[0]) / cell).astype(int), 0, ny - 1)
        depth = float(np.max(M[ii, jj] - sz))
        if depth > allow:
            n_bury += 1
            k = int(np.argmax(M[ii, jj] - sz))
            if bury_loc is None or depth - allow > bury_loc[4]:
                bury_loc = (float(sx[k]), float(sy[k]), float(sz[k]),
                            ln_no, depth - allow)
            w_bury = max(w_bury, depth - allow)
        zmin_move = float(sz.min())
        if np.isfinite(zmax) and zmax > zmin_move + Bc:
            r_max = max((zmax - zmin_move + Bc) / max(tan_t, 1e-6),
                        head_r if zmax - zmin_move > head_h else 0.0
                        ) + 2 * cell
            j0 = max(int((sx.min() - r_max - xs[0]) / cell), 0)
            j1 = min(int((sx.max() + r_max - xs[0]) / cell) + 2, nx)
            i0 = max(int((sy.min() - r_max - ys[0]) / cell), 0)
            i1 = min(int((sy.max() + r_max - ys[0]) / cell) + 2, ny)
            Mw = M[i0:i1, j0:j1]
            fin = np.isfinite(Mw)
            if fin.any():
                dd = np.hypot(sx[:, None] - XX[i0:i1, j0:j1][fin][None, :],
                              sy[:, None] - YY[i0:i1, j0:j1][fin][None, :])
                Mwf = Mw[fin][None, :]
                intr_cone = Mwf - (sz[:, None] + tan_t * dd + Bc)
                intr_head = np.where(dd < head_r,
                                     Mwf - (sz[:, None] + head_h),
                                     -np.inf)
                intr = np.maximum(intr_cone, intr_head)
                mi = float(intr.max())
                if mi > 0:
                    per = intr.max(axis=1)
                    n_cone += int((per > 0).sum())
                    w_cone = max(w_cone, mi)
                    if cone_loc is None or mi > cone_loc[4]:
                        k = int(np.argmax(per))
                        cone_loc = (float(sx[k]), float(sy[k]),
                                    float(sz[k]), ln_no, mi)
        if extr:
            np.maximum.at(M, (ii, jj), sz)
            zmax = max(zmax, float(sz.max()))
    ok_bc = n_bury == 0 and n_cone == 0
    print(f"[burial]  {n_bury} (worst +{w_bury:.2f}mm, allowance "
          f"{allow:.2f}mm)  {'PASS' if n_bury == 0 else 'FAIL'}")
    print(f"[cone]    {n_cone} intrusions incl. housing floor "
          f"(worst +{w_cone:.2f}mm)  {'PASS' if n_cone == 0 else 'FAIL'}")
    if bury_loc:
        print(f"  worst burial at X={bury_loc[0]:.1f} Y={bury_loc[1]:.1f} "
              f"Z={bury_loc[2]:.2f} (line {bury_loc[3]}, "
              f"+{bury_loc[4]:.2f}mm)")
    if cone_loc:
        print(f"  worst cone/housing hit at X={cone_loc[0]:.1f} "
              f"Y={cone_loc[1]:.1f} Z={cone_loc[2]:.2f} "
              f"(line {cone_loc[3]}, +{cone_loc[4]:.2f}mm)")

    # --- pass 2: lateral flat-tip gouge --------------------------------
    M2 = np.full((ny, nx), -np.inf)
    d_min, d_max = 0.15, max(tip_r, line_w) + 0.35
    R = int(math.ceil(d_max / cell))
    off = [(di, dj, math.hypot(di * cell, dj * cell))
           for di in range(-R, R + 1) for dj in range(-R, R + 1)
           if d_min <= math.hypot(di * cell, dj * cell) <= d_max]
    odi = np.array([o[0] for o in off])
    odj = np.array([o[1] for o in off])
    tan_cone = math.tan(math.radians(cone_deg))
    # grid-quantization allowance: heights are cell maxima, so on a
    # layer sloped at theta a legitimate neighbour can APPEAR up to
    # cell_diag*tan(theta) taller than it is at the sample's true
    # distance — same reasoning as [burial]'s slope allowance.
    quant = math.hypot(cell, cell) * math.tan(math.radians(theta))
    clr = gouge_tol + quant + np.maximum(
        np.array([o[2] for o in off]) - tip_r, 0.0) * tan_cone
    n_lat = 0
    lat_worst = 0.0
    lat_loc = None
    same_surface = 0.6
    for (ln_no, x0, y0, z0, x1, y1, z1, extr) in iter_moves(
            args.gcode, bed_x, bed_y, layer_mark, body_end_mark):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(L / 0.3) + 1)
        ts = np.linspace(0, 1, n)
        sx = x0 + ts * (x1 - x0)
        sy = y0 + ts * (y1 - y0)
        sz = z0 + ts * (z1 - z0)
        ii = np.clip(((sy - ys[0]) / cell).astype(int), 0, ny - 1)
        jj = np.clip(((sx - xs[0]) / cell).astype(int), 0, nx - 1)
        ni = np.clip(ii[:, None] + odi[None, :], 0, ny - 1)
        nj = np.clip(jj[:, None] + odj[None, :], 0, nx - 1)
        pen = M2[ni, nj] - sz[:, None]
        sev = np.where(pen < same_surface, pen - clr[None, :], -np.inf)
        per = sev.max(axis=1)
        hit = per > 0.005
        if hit.any():
            n_lat += int(hit.sum())
            k = int(np.argmax(per))
            if per[k] > lat_worst:
                lat_worst = float(per[k])
                lat_loc = (round(float(sx[k]), 1), round(float(sy[k]), 1),
                           round(float(sz[k]), 2), ln_no)
        if extr:
            np.maximum.at(M2, (ii, jj), sz)
    ok_lat = lat_worst <= 0.02
    lat_status = ("PASS" if ok_lat else
                  ("WARN (slope override)" if slope_override else "FAIL"))
    print(f"[lateral] flat-tip gouge: {n_lat} beyond tolerance "
          f"(worst +{lat_worst:.3f}mm past tol)  {lat_status}")
    if lat_loc:
        print(f"  worst gouge at X={lat_loc[0]} Y={lat_loc[1]} "
              f"Z={lat_loc[2]} (line {lat_loc[3]})")
    if not ok_lat and slope_override:
        print("  NOTE: --max_slope override active — lateral tip-gouge is "
              "a WARNING, not a failure. Expect surface marks where the "
              "flat tip plows adjacent beads on the slope; inspect the "
              "print. (floor/burial/cone stay enforced.)")

    # collision gates (floor/burial/cone) always hard; the lateral
    # tip-gouge is a SURFACE-QUALITY gate, softened when the user has
    # explicitly overridden the slope cap.
    if ok_floor and ok_bc and (ok_lat or slope_override):
        if ok_lat:
            print("VERIFICATION PASSED")
        else:
            print("VERIFICATION PASSED (with lateral tip-gouge "
                  "WARNING — slope override active)")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
