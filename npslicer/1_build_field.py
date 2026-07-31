"""
NP-Slicer — stage 1: build the warp field from ANY STL
=========================================================
Input:  an STL (arbitrary geometry) + a printer profile.
Output: <stem>_field.npz (the warp map) + <stem>_field.png (preview).

There are TWO field modes; the tool picks one automatically (--mode auto):

  * shear — S(x,y) = the part's BOTTOM
    height field, gradient-limited, bounded_extend across the
    silhouette. Layers follow the S-curve through the WHOLE thickness
    (stress-aligned deposition) — the right mode for constant-
    thickness swept parts (the task dogbone). Pure shear: thickness
    factor == 1, no blend zone. Raised undersides get grid supports
    spliced in stage 3; PrusaSlicer supports must be OFF.
  * top   — top-surface displacement,
    depth-decayed to a flat bottom; supports come from PrusaSlicer.

  auto rule: shear when the part is sweep-like — over >= 70% of the
  footprint the thickness (T - B) is constant within ~0.5 mm AND the
  underside actually rises (max B > 1 mm). Decision + reason are
  printed as a "MODE:" line (the web UI shows it, with an override).

There is no GLOBAL clearance clamp: a naive D_range <= clearance cap
would hold a 7 mm S-rise to 3.5 mm and print stairs on half the ramp.
The physical constraint is WINDOWED: the rise within the printhead
footprint radius, not the total rise. Since |grad field| <= tan_field,
the rise within radius r is <= tan_field*r, so the builder only checks
    tan_field * shroud_radius <= nonplanar_clearance_h   (near field)
    tan_field * head_radius   <= head_clearance_h        (far field)
and scales tan_field down (with a warning) only if violated. The
stage-4 verifier still checks the true windowed constraints temporally
— safety stays enforced independent of the builder.

Degenerate cases report PLANAR PASSTHROUGH instead of inventing a warp:
flat-topped parts (cube), and in auto mode parts where warping has
near-zero benefit (slant: its 35-45 deg face exceeds any realistic
slope cap; those stairs are 3-axis physics, not something to fight).

Run:
  python 1_build_field.py --stl inputs/stls/wave.stl --printer prusa_mk4
"""

import argparse
import math
import os
import sys

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as vc                                    # noqa: E402


class FieldSlopeError(RuntimeError):
    """Silhouette self-check failure (open issue #1). Signals the driver
    to auto-retry with a coarser grid instead of aborting — this trips
    more at a fine grid with a low slope cap (e.g. the P1S 6 deg tip)."""


def main():
    ap = argparse.ArgumentParser(description="stage 1: build field")
    ap.add_argument("--stl", required=True)
    ap.add_argument("--printer", default="prusa_mk4",
                    help="Profile name in npslicer/profiles/ or a .json path")
    ap.add_argument("--mode", choices=["auto", "shear", "top"],
                    default="auto",
                    help="Warp mode; auto picks shear for sweep-like "
                         "parts (constant thickness, raised underside), "
                         "top otherwise")
    ap.add_argument("--theta_target", type=float, default=None,
                    help="Max layer slope (deg), QUALITY preference; "
                         "always clamped by the hardware caps from the "
                         "profile. Default: profile theta_target_default")
    ap.add_argument("--max_slope", type=float, default=None,
                    help="EXPERIMENTAL OVERRIDE: force this max layer "
                         "slope (deg), BYPASSING the lateral tip-gouge "
                         "and cone quality caps. Windowed-clearance "
                         "collision safety STILL applies, and the stage-4 "
                         "verifier downgrades the lateral tip-gouge check "
                         "to a non-fatal warning. Default: off (caps "
                         "enforced — normal behaviour).")
    ap.add_argument("--grid_step", type=float, default=0.35,
                    help="Field grid step (mm), isotropic")
    ap.add_argument("--grid_max", type=int, default=500,
                    help="Max grid nodes per axis")
    ap.add_argument("--blend_factor", type=float, default=2.5,
                    help="[top mode] d_blend = blend_factor * max "
                         "displacement (>= 1.25 for monotonicity)")
    ap.add_argument("--blend_min", type=float, default=3.0,
                    help="[top mode] Minimum blend depth (mm)")
    ap.add_argument("--keep_flat_bottom", type=float, default=1.0,
                    help="[top mode] Never bend layers below this "
                         "height (mm)")
    ap.add_argument("--clearance", type=float, default=None,
                    help="Override profile nonplanar_clearance_h (mm)")
    ap.add_argument("--min_benefit", type=float, default=0.05,
                    help="[auto+top] planar passthrough if less than "
                         "this fraction of the footprint is conformable "
                         "AND meaningfully displaced (D > 0.3 mm)")
    ap.add_argument("--center_xy", default=None,
                    help="Where the part center lands on the bed 'X,Y'; "
                         "default = bed middle")
    ap.add_argument("--line_width", type=float, default=None)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    _run(args)


def _run(args, _attempt=0):
    prof = vc.load_printer_profile(args.printer)
    bed_x, bed_y = float(prof["bed_x"]), float(prof["bed_y"])
    cone_deg = float(prof["nozzle_cone_angle_deg"])
    tip_r = float(prof["tip_land_radius"])
    gouge_tol = float(prof["gouge_tol_mm"])
    clearance = (args.clearance if args.clearance is not None
                 else float(prof["nonplanar_clearance_h"]))
    head_h = float(prof["head_clearance_h"])
    head_r = float(prof["head_radius"])
    shroud_r = float(prof.get("shroud_radius", 8.0))
    if prof.get("shroud_radius") is None:
        print("NOTE: profile has no shroud_radius — assuming 8.0 mm "
              "(add the key after measuring the toolhead).")
    line_w = (args.line_width if args.line_width is not None
              else float(prof["nozzle_diameter"]) * 1.125)
    theta_q = (args.theta_target if args.theta_target is not None
               else float(prof.get("theta_target_default", 20.0)))
    lateral_deg, cap_src = vc.lateral_slope_cap_deg(
        tip_r, line_w, cone_deg, gouge_tol)
    slope_override = args.max_slope is not None
    if slope_override:
        theta = float(args.max_slope)
        print(f"NOTE: --max_slope {theta:.1f} deg OVERRIDES the hardware "
              f"quality caps (lateral tip {lateral_deg:.1f} deg, cone "
              f"{cone_deg:.1f} deg). Windowed-clearance collision safety "
              f"still applies; the stage-4 verifier will DOWNGRADE the "
              f"lateral tip-gouge check to a warning. Experimental — "
              f"inspect the print for surface marks.")
    else:
        theta = min(theta_q, cone_deg, lateral_deg)
        if theta < theta_q:
            driver = ("lateral tip cap" if lateral_deg <= cone_deg
                      else "nozzle cone")
            print(f"NOTE: theta_target {theta_q:.1f} deg clamped to "
                  f"{theta:.1f} deg by the {driver} ({cap_src}). Edit the "
                  f"profile after measuring the real toolhead.")
    tan_max = float(np.tan(np.radians(theta)))
    # the grid caps PER-AXIS node differences, but the bilinear surface
    # between nodes combines x- and y-gradients: worst-case |grad| is
    # sqrt(2) times the per-axis bound. Erode with tan/sqrt(2) so the
    # INTERPOLATED field (what the nozzle actually rides) never exceeds
    # the physical cap — verified end-to-end by the verifier's lateral check.
    tan_field = tan_max / math.sqrt(2.0)

    # --- WINDOWED clearance check (replaces a global rise clamp) ---
    # rise within radius r of the tip is <= tan_field * r; only clamp
    # the field slope if that windowed rise would exceed a clearance.
    tan_win = min(clearance / max(shroud_r, 1e-6),
                  head_h / max(head_r, 1e-6))
    if tan_field > tan_win + 1e-9:
        print(f"WARNING: windowed clearance clamps the field slope "
              f"{math.degrees(math.atan(tan_field)):.1f} -> "
              f"{math.degrees(math.atan(tan_win)):.1f} deg "
              f"(rise within shroud r={shroud_r:.0f} mm must stay "
              f"<= {clearance:.1f} mm, within head r={head_r:.0f} mm "
              f"<= {head_h:.1f} mm).")
        tan_field = tan_win
    print(f"Printer: {prof['name']}  cone {cone_deg:.1f} deg  "
          f"tip_r {tip_r:.2f} mm  lateral cap {lateral_deg:.1f} deg  "
          f"->  theta = {theta:.1f} deg "
          f"(grid erosion at {math.degrees(math.atan(tan_field)):.1f} deg)")
    print(f"Windowed clearance: rise within shroud r={shroud_r:.0f} mm "
          f"= {tan_field*shroud_r:.2f} <= {clearance:.1f} mm, within "
          f"head r={head_r:.0f} mm = {tan_field*head_r:.2f} <= "
          f"{head_h:.1f} mm  OK (windowed, no global rise cap)")

    # --- mesh: center on bed, drop to Z=0 ----------------------------
    mesh = vc.load_mesh(args.stl)
    if args.center_xy:
        cx, cy = (float(v) for v in args.center_xy.split(","))
    else:
        cx, cy = bed_x / 2.0, bed_y / 2.0
    mid = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    translation = [cx - mid[0], cy - mid[1], -float(mesh.bounds[0, 2])]
    mesh.apply_translation(translation)
    ex = float(mesh.bounds[1, 0] - mesh.bounds[0, 0])
    ey = float(mesh.bounds[1, 1] - mesh.bounds[0, 1])
    part_h = float(mesh.bounds[1, 2])
    print(f"Part: {ex:.1f} x {ey:.1f} x {part_h:.1f} mm at "
          f"({cx:.1f}, {cy:.1f})")

    # --- top/bottom height maps ---------------------------------------
    margin = 3.0
    nx = int(np.clip(round((ex + 2 * margin) / args.grid_step) + 1,
                     24, args.grid_max))
    ny = int(np.clip(round((ey + 2 * margin) / args.grid_step) + 1,
                     24, args.grid_max))
    xs = np.linspace(mesh.bounds[0, 0] - margin,
                     mesh.bounds[1, 0] + margin, nx)
    ys = np.linspace(mesh.bounds[0, 1] - margin,
                     mesh.bounds[1, 1] + margin, ny)
    print(f"Grid: {nx} x {ny} nodes "
          f"(step {(xs[-1]-xs[0])/(nx-1):.3f} x "
          f"{(ys[-1]-ys[0])/(ny-1):.3f} mm)")
    xx, yy = np.meshgrid(xs, ys)
    origins = np.column_stack([xx.ravel(), yy.ravel(),
                               np.full(nx * ny, -1.0)])
    dirs = np.tile([0.0, 0.0, 1.0], (len(origins), 1))
    hits, idx, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True)
    T = np.full((ny, nx), np.nan)
    B = np.full((ny, nx), np.nan)
    n_hits = np.zeros(nx * ny, dtype=int)
    if len(hits):
        np.fmax.at(T, (idx // nx, idx % nx), hits[:, 2])
        np.fmin.at(B, (idx // nx, idx % nx), hits[:, 2])
        np.add.at(n_hits, idx, 1)
    valid = ~np.isnan(T)
    if not valid.any():
        raise SystemExit("ERROR: no rays hit the mesh — corrupt STL?")
    # mask repair: vertical ray casting speckles the mask at walls/edges
    # (near-tangent misses). A ragged edge makes flat_extend mix donors
    # from different rim points -> silhouette slope spikes (issue #1).
    closed = ndi.binary_closing(valid, structure=np.ones((3, 3)))
    holes = closed & ~valid
    if holes.any():
        Tf = np.where(valid, T, -np.inf)
        Bf = np.where(valid, B, np.inf)
        for _ in range(3):
            Tn = ndi.grey_dilation(Tf, size=(3, 3))
            Tf = np.where(np.isfinite(Tf), Tf, Tn)
            Bn = -ndi.grey_dilation(-Bf, size=(3, 3))
            Bf = np.where(np.isfinite(Bf), Bf, Bn)
        T = np.where(holes & np.isfinite(Tf), Tf, T)
        B = np.where(holes & np.isfinite(Bf), Bf, B)
        print(f"Mask repair: {int(holes.sum())} ray-miss cells closed")
        valid = closed & ~np.isnan(T)
    multi = (n_hits.reshape(ny, nx) > 2) & valid
    frac_multi = multi.sum() / max(valid.sum(), 1)
    print(f"Columns: {int(valid.sum())} part, "
          f"{frac_multi*100:.1f}% with cavities/undercuts")
    if frac_multi > 0.05:
        print("WARNING: many multi-interval columns. The TOP surface "
              "still drives the top field correctly, and the shear "
              "field uses only the outermost bottom — but internal "
              "cavities would be bridged; check the preview image.")

    # --- mode decision ------------------------------------------------
    thick = (T - B)[valid]
    t_med = float(np.median(thick))
    frac_const = float((np.abs(thick - t_med) <= 0.5).mean())
    maxB = float(B[valid].max())
    sweep_like = (frac_const >= 0.70) and (maxB > 1.0)
    detail = (f"{frac_const*100:.0f}% of footprint at thickness "
              f"{t_med:.2f} mm (+-0.5), max bottom rise {maxB:.2f} mm")
    if args.mode == "auto":
        mode = "shear" if sweep_like else "top"
        why = (f"auto: sweep-like — {detail}" if sweep_like else
               f"auto: not sweep-like — {detail}")
    else:
        mode = args.mode
        why = f"forced by --mode (auto would say " \
              f"{'shear' if sweep_like else 'top'}: {detail})"
    print(f"MODE: {mode} ({why})")

    fpath, stem = vc.field_path(args.outdir, args.stl)
    meta_common = {
        "stl": os.path.abspath(args.stl),
        "translation": translation,
        "mode": mode, "mode_reason": why,
        "theta_deg": theta,
        "theta_quality": theta_q,
        "slope_override": slope_override,
        "cone_deg": cone_deg,
        "lateral_cap_deg": lateral_deg,
        "tip_land_radius": tip_r,
        "gouge_tol": gouge_tol,
        "line_width": line_w,
        "nozzle_diameter": float(prof["nozzle_diameter"]),
        "clearance": clearance,
        "shroud_radius": shroud_r,
        "head_clearance_h": head_h,
        "head_radius": head_r,
        "bed_x": bed_x, "bed_y": bed_y,
        "part_height": part_h,
        "profile": prof["_path"],
    }

    try:
        if mode == "shear":
            build_shear(args, xs, ys, T, B, valid, tan_field, theta,
                        meta_common, fpath, stem)
        else:
            build_top(args, xs, ys, T, B, valid, tan_field, theta,
                      meta_common, fpath, stem, part_h)
    except FieldSlopeError as e:
        # open issue #1: the silhouette field slope exceeds the cap at
        # this grid. Auto-coarsen and retry so the DEFAULT run works
        # instead of erroring (a coarser grid averages out the rim spike).
        if _attempt >= 4:
            raise SystemExit(
                f"{e}\nAuto-coarsening reached grid_step "
                f"{args.grid_step:.2f} mm and still failed — please "
                f"report this model.")
        nxt = round(args.grid_step * 1.4, 3)
        print(f"NOTE: silhouette self-check failed at grid_step "
              f"{args.grid_step:.2f} mm; auto-retrying coarser at "
              f"{nxt:.2f} mm (issue #1 worsens at fine grid + low cap).")
        args.grid_step = nxt
        return _run(args, _attempt + 1)


# ---------------------------------------------------------------------
def preview(outdir, stem, xs, ys, panels):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        ext = [xs[0], xs[-1], ys[0], ys[-1]]
        for ax, (data, title) in zip(axes, panels):
            im = ax.imshow(data, origin="lower", extent=ext,
                           cmap="plasma")
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
        plt.tight_layout()
        png = os.path.join(outdir, f"{stem}_field.png")
        plt.savefig(png, dpi=140)
        print(f"Preview -> {png}")
    except Exception as e:                        # matplotlib optional
        print(f"(preview skipped: {e})")


def slope_selfcheck(F, xs, ys, tan_field, theta):
    chk = vc.grid_slope_deg(F, xs, ys)
    theta_field = math.degrees(math.atan(tan_field))
    print(f"Field self-check: max node-to-node slope {chk:.1f} deg "
          f"(erosion cap {theta_field:.1f} deg, physical cap "
          f"{theta:.1f} deg)")
    if chk > theta_field * 1.25 + 1.0:
        raise FieldSlopeError(
            f"field slope {chk:.1f} deg exceeds the cap after silhouette "
            f"extension (open issue #1)")


# ---------------------------------------------------------------------
def build_shear(args, xs, ys, T, B, valid, tan_field, theta,
                meta, fpath, stem):
    """Bottom-shear field. Hard-won field rules:
      * erosion seeded by PART COLUMNS ONLY — filling the outside first
        dragged S below the true bottom across rigid flat regions,
        tilting them in warped space (terraced rims);
      * S == B wherever the slope cap allows: rigid regions warp by
        pure translation and slice cleanly;
      * flat continuation outside the silhouette (bounded_extend keeps
        the field flat across every silhouette edge)."""
    S_seed = np.where(valid, B, np.inf)
    S = vc.gradient_limit(S_seed, xs, ys, tan_field)
    if not np.isfinite(S).all():
        raise SystemExit("ERROR: shear-field erosion did not converge.")
    dev = np.abs(S - B)[valid]
    print(f"Shear field: S == bottom on {(dev < 0.05).mean()*100:.0f}% "
          f"of part columns; max relaxation {dev.max():.2f} mm "
          f"(underside steeper than the cap there)")
    S = vc.bounded_extend(S, valid, xs, ys, tan_field)
    S = np.maximum(S, 0.0)
    slope_selfcheck(S, xs, ys, tan_field, theta)
    S_range = float(S[valid].max())

    passthrough = S_range < 0.30
    if passthrough:
        print(f"PLANAR PASSTHROUGH: bottom relief is only "
              f"{S_range:.2f} mm (< 0.30) — the underside is "
              f"effectively flat; shear would do nothing.")
        S[:] = 0.0

    # anchor cells: S == B -> the warped bottom lands exactly on the
    # bed plane. Stage 2 checks the warped mesh against this (the shear
    # equivalent of the top-flatness prediction check).
    anchor_cells = valid & (np.abs(S - B) < 0.05)
    conf_frac = anchor_cells.sum() / max(valid.sum(), 1)
    W_expect_bottom = B - S            # warped bottom per column

    # supports: raised underside floats in real space
    sup_mask = valid & (B > 0.5) & ~passthrough
    print(f"Support columns (raised underside): {int(sup_mask.sum())}"
          + ("" if sup_mask.any() else " — no supports needed"))

    meta.update(passthrough=bool(passthrough))
    sf = vc.ShearField(xs, ys, S, meta)
    rt = sf.self_test_roundtrip()
    print(f"Self-test: warp/unwarp round-trip worst error {rt:.2e} mm")
    sf.save(fpath, T=np.where(valid, T, np.nan),
            B=np.where(valid, B, np.nan), valid=valid,
            sup_mask=sup_mask, anchor_cells=anchor_cells,
            W_expect_bottom=W_expect_bottom)

    preview(args.outdir, stem, xs, ys, [
        (np.where(valid, T, np.nan), "Top surface T"),
        (np.where(valid, B, np.nan), "Bottom surface B (raised = "
                                     "gets NP supports)"),
        (S, f"Shear field S (max {S_range:.2f} mm) — layers follow "
            f"this")])

    print(f"\nField -> {fpath}")
    print(f"Summary: mode=SHEAR  S_max={S_range:.2f} mm  "
          f"S==B on {conf_frac*100:.0f}% of columns  "
          f"supports={'yes' if sup_mask.any() else 'no'}"
          + ("  PLANAR PASSTHROUGH" if passthrough else ""))
    print("REMINDER: slice the warped STL with PrusaSlicer supports "
          "OFF — stage 3 splices its own NP supports for the raised "
          "underside; slicer supports would collide with them.")
    print(f"Next: python 2_warp_stl.py --field {fpath}")


# ---------------------------------------------------------------------
def build_top(args, xs, ys, T, B, valid, tan_field, theta,
              meta, fpath, stem, part_h):
    """Top-surface depth-decayed field:
    no global clearance clamp (windowed check already done), and the
    anchor-selection cap term generalized (see comment below)."""
    tan_max = math.tan(math.radians(theta))

    # --- local top analysis: slope + grid-scale smoothness -------------
    _, (iyn, ixn) = ndi.distance_transform_edt(~valid,
                                               return_indices=True)
    T_near = np.where(valid, T, T[iyn, ixn])      # nearest-fill, no NaN
    T_mean = ndi.uniform_filter(T_near, size=3)
    rough = np.abs(T_near - T_mean) > 0.08
    gyT, gxT = np.gradient(T_near, ys, xs)
    slopeT = np.hypot(gxT, gyT)
    eligible = valid & (slopeT <= tan_max) & ~rough

    # --- the printable top field T~ -----------------------------------
    # erosion A: seeded by ALL part columns (the safe baseline)
    T_lim = vc.gradient_limit(np.where(valid, T, np.inf), xs, ys,
                              tan_field)
    if not np.isfinite(T_lim).all():
        raise SystemExit("ERROR: gradient limiting did not converge.")
    # dilation B: eligible top cells extended DOWNHILL at the slope cap
    # (keeps a high conformable island conformal instead
    # of letting far-away steep sides erode it down)
    if eligible.any():
        T2 = -vc.gradient_limit(np.where(eligible, -T, np.inf), xs, ys,
                                tan_field)
        T_lim = np.maximum(T_lim, T2)
    H = float(T_lim[valid].max())

    # --- anchor selection --------------------------------------------
    # There is no global CLEARANCE cap to score anchors against. What
    # still limits a candidate anchor Hc
    # is MONOTONICITY: displacements beyond MONO_RATIO*(Hc -
    # keep_flat_bottom) force short-part scaling that flattens the
    # surface. Score with that cap instead (Prop: a hub anchor would
    # force scaling that flattens the blades; the selector still
    # returns the anchor to the blades, hub prints planar).
    lab, nl = ndi.label(eligible)
    cands = {H}
    if nl:
        sizes = ndi.sum(np.ones_like(lab), lab, range(1, nl + 1))
        order = np.argsort(sizes)[::-1][:8]      # 8 largest islands
        for i in order:
            cands.add(float(T[lab == i + 1].max()))
    best_H, best_n = H, -1
    for Hc in sorted(cands):
        cap_c = max(vc.WarpField.MONO_RATIO
                    * (Hc - args.keep_flat_bottom), 0.1)
        T_eff = np.minimum(np.maximum(T_lim, Hc - cap_c), Hc)
        n_conf = int((valid & (np.abs(T_eff - T) < 0.15) & ~rough).sum())
        if n_conf > best_n:
            best_H, best_n = Hc, n_conf
    if best_H != H:
        print(f"Anchor selection: H {H:.2f} -> {best_H:.2f} mm "
              f"(maximizes conformal coverage; higher features print "
              f"planar)")
    H = best_H
    T_lim = np.minimum(T_lim, H)
    # (no global head-clearance clamp on D here — the windowed check in
    # main() covers it, and stage 4 verifies the real thing.)

    # monotonicity clamp, GRACEFUL: a monotone map for this part
    # can realize at most MONO_RATIO * (H - keep_flat_bottom) of
    # displacement. Rather than scaling the whole field down (which would
    # flatten even the shallow, conformable regions), the limit is
    # clamped per column — deep regions keep planar stairs (graceful
    # degradation), shallow regions stay conformal.
    d_max = H - args.keep_flat_bottom
    if d_max <= 0:
        raise SystemExit("ERROR: part shorter than keep_flat_bottom.")
    need_max = vc.WarpField.MONO_RATIO * d_max
    D_pre = H - T_lim
    if float(D_pre[valid].max()) > need_max:
        print(f"NOTE: displacement {D_pre[valid].max():.2f} mm exceeds "
              f"the monotone-map limit {need_max:.2f} mm for this part "
              f"height -> clamped; deeper regions keep planar stairs "
              f"(graceful degradation).")
        T_lim = np.maximum(T_lim, H - need_max)

    T_lim = vc.bounded_extend(T_lim, valid, xs, ys, tan_field)
    D = np.maximum(H - T_lim, 0.0)
    slope_selfcheck(D, xs, ys, tan_field, theta)
    D_range = float(D[valid].max())

    # --- passthrough decisions -----------------------------------------
    conformable_pre = valid & (np.abs((H - D) - T) < 0.15) & ~rough
    benefit = conformable_pre & (D > 0.30)
    benefit_frac = float(benefit.sum() / max(valid.sum(), 1))
    passthrough = D_range < 0.30
    if passthrough:
        print(f"PLANAR PASSTHROUGH: top relief is only {D_range:.2f} mm "
              f"(< 0.30). The part is effectively flat-topped — the "
              f"pipeline will keep the G-code fully planar (correct "
              f"behaviour, e.g. for a cube).")
    elif (args.mode == "auto"
          and benefit_frac < args.min_benefit):
        # near-zero benefit: almost nowhere does the field both track
        # the true top AND move material (slant: the 35-45 deg face
        # exceeds any realistic cap; bending layers toward the top cap
        # would 'curve where the geometry is straight')
        passthrough = True
        print(f"PLANAR PASSTHROUGH (near-zero benefit): only "
              f"{benefit_frac*100:.1f}% of the footprint is "
              f"conformable AND displaced > 0.3 mm "
              f"(< {args.min_benefit*100:.0f}%). Force --mode top to "
              f"warp anyway.")
    else:
        print(f"Benefit check: {benefit_frac*100:.1f}% of footprint "
              f"conformable and displaced (>= "
              f"{args.min_benefit*100:.0f}% needed in auto mode)")
    if passthrough:
        D[:] = 0.0
        D_range = 0.0
        d_blend = max(args.blend_min, 1.0)
    else:
        # D_range <= need_max = MONO_RATIO * d_max by the clamp above,
        # so monotonicity holds without whole-field scaling
        d_blend = min(max(args.blend_factor * D_range, args.blend_min),
                      d_max)

    meta.update(passthrough=bool(passthrough),
                benefit_frac=benefit_frac)
    wf = vc.WarpField(xs, ys, D, H, d_blend, meta)
    rt = wf.self_test_roundtrip()
    print(f"Self-test: warp/unwarp round-trip worst error {rt:.2e} mm")

    # conformable = final field tracks the true top (after any scaling
    # of D) and the top is grid-smooth; W_expect = the field's own
    # prediction of the warped top per column (stage-2 flatness check)
    _, (iyn, ixn) = ndi.distance_transform_edt(~valid,
                                               return_indices=True)
    T_near = np.where(valid, T, T[iyn, ixn])
    T_mean = ndi.uniform_filter(T_near, size=3)
    rough = np.abs(T_near - T_mean) > 0.08
    conformable = valid & (np.abs((H - D) - T) < 0.15) & ~rough
    conf_frac = conformable.sum() / max(valid.sum(), 1)
    XXg, YYg = np.meshgrid(xs, ys)
    W_expect = wf.z_warped(XXg.ravel(), YYg.ravel(),
                           T_near.ravel()).reshape(T_near.shape)

    wf.save(fpath, T=np.where(valid, T, np.nan),
            B=np.where(valid, B, np.nan), valid=valid,
            T_lim=T_lim, conformable=conformable, W_expect=W_expect)

    preview(args.outdir, stem, xs, ys, [
        (np.where(valid, T, np.nan), "Top surface T"),
        (T_lim, "Printable top T~ (slope-limited)"),
        (D, f"Displacement D (max {D_range:.2f} mm, "
            f"blend {d_blend:.2f} mm)")])

    print(f"\nField -> {fpath}")
    print(f"Summary: mode=TOP  H={H:.2f} mm  D_max={D_range:.2f} mm  "
          f"d_blend={d_blend:.2f} mm  conformable top area "
          f"{conf_frac*100:.0f}%  "
          f"{'PLANAR PASSTHROUGH' if passthrough else 'NON-PLANAR'}")
    print(f"Next: python 2_warp_stl.py --field {fpath}")


if __name__ == "__main__":
    main()
