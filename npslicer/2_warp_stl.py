"""
NP-Slicer — stage 2: warp the STL for planar slicing
=======================================================
Loads the field from stage 1, applies the forward warp to the
(uniformly subdivided) mesh and REFUSES to emit a bad file:

  * uniform subdivision only — adaptive subdivide_to_size produces
    T-junction leaks;
  * FLATNESS SELF-CHECK, per mode: top mode — the printable top
    region must become FLAT at height H in warped space; shear mode —
    the warped BOTTOM must sit on the Z=0 plane over every column
    where S == B. Either failing is the 'warped STL not flat' class of
    bug: this stage fails loudly instead of letting you slice garbage;
  * watertightness check on the result.

Then slice outputs/<stem>_warped.stl in the PrusaSlicer GUI:
  - do NOT move / rotate / scale the object;
  - 'Use relative E distances' ON (MK4 default), Arc fitting OFF,
    wipe tower OFF;
  - SHEAR mode only: PrusaSlicer supports OFF (the warped part sits
    flat — the slicer sees nothing to support; stage 3 splices its
    own NP supports for the raised underside);
  - export G-code and continue with 3_unwarp_gcode.py.

Run:
  python 2_warp_stl.py --field outputs/wave_field.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as vc                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="stage 2: warp STL")
    ap.add_argument("--field", required=True,
                    help="<stem>_field.npz from 1_build_field.py")
    ap.add_argument("--stl", default=None,
                    help="Override the STL recorded in the field file")
    ap.add_argument("--max_edge", type=float, default=1.5,
                    help="Uniform-subdivision target edge length (mm)")
    ap.add_argument("--outdir", default=None,
                    help="Default: the field file's folder")
    args = ap.parse_args()

    import trimesh

    wf = vc.load_field(args.field)
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.field))
    stl = args.stl or wf.meta["stl"]
    stem = os.path.splitext(os.path.basename(args.field))[0]
    stem = stem[:-6] if stem.endswith("_field") else stem

    mesh = vc.load_mesh(stl)
    mesh.apply_translation(wf.meta["translation"])

    if wf.meta.get("passthrough"):
        wpath = os.path.join(outdir, f"{stem}_warped.stl")
        mesh.export(wpath)
        print(f"PLANAR PASSTHROUGH: exported the (positioned) original "
              f"mesh unchanged -> {wpath} — see the stage-1 log for "
              f"the reason (flat top, flat underside in shear mode, "
              f"or near-zero benefit)")
        print("Slice it planar as usual; the unwarp stage will also "
              "pass the G-code through unchanged.")
        return

    # --- conforming adaptive refinement (Rivara bisection) -----------
    # replaces plain uniform subdivision, which either leaked
    # (subdivide_to_size) or exploded to 23M faces / OOM on meshes with
    # mixed edge sizes (Prop.stl: 38 mm radial edges next to 0.5 mm rim)
    V, F = vc.refine_mesh(mesh.vertices, mesh.faces, args.max_edge)
    sub = trimesh.Trimesh(V, F, process=True)
    print(f"Refined: {len(sub.faces)} faces "
          f"(from {len(mesh.faces)}, watertight={sub.is_watertight})")
    if not sub.is_watertight and mesh.is_watertight:
        raise SystemExit("ERROR: subdivision broke watertightness "
                         "(unexpected — please report this model).")

    # --- forward warp ---------------------------------------------------
    Vw = sub.vertices.copy()
    Vw[:, 2] = wf.z_warped(Vw[:, 0], Vw[:, 1], Vw[:, 2])
    n_neg = int((Vw[:, 2] < 0).sum())
    if n_neg:
        print(f"  ({n_neg} vertices clamped to bed, worst "
              f"{-Vw[:, 2].min():.3f} mm interpolation noise)")
    Vw[:, 2] = np.maximum(Vw[:, 2], 0.0)
    warped = trimesh.Trimesh(Vw, sub.faces, process=True)

    # --- FLATNESS SELF-CHECK ---------------------------------------------
    # Regression guard for the "warping step returned a non-flat STL"
    # failure, per mode:
    #   top:   warped TOP over conformable cells must sit at the
    #          field's own prediction (W_expect, ~H);
    #   shear: warped BOTTOM over anchor cells (S == B) must sit on
    #          the Z=0 plane (W_expect_bottom, == 0 there).
    if wf.mode == "shear":
        conf = wf.raw["anchor_cells"]
        expect_grid = wf.raw["W_expect_bottom"]
        use_top = False
        what = "warped-bottom-on-bed"
    else:
        conf = wf.raw["conformable"]
        expect_grid = wf.raw["W_expect"]
        use_top = True
        what = "warped-top flatness"
    xs, ys = wf.xs, wf.ys
    ii, jj = np.where(conf)
    if len(ii):
        if len(ii) > 4000:
            sel = np.linspace(0, len(ii) - 1, 4000).astype(int)
            ii, jj = ii[sel], jj[sel]
        org = np.column_stack([xs[jj], ys[ii], np.full(len(ii), -1.0)])
        drc = np.tile([0.0, 0.0, 1.0], (len(org), 1))
        hits, hidx, _ = warped.ray.intersects_location(
            org, drc, multiple_hits=True)
        surf = np.full(len(ii), np.nan)
        if len(hits):
            if use_top:
                np.fmax.at(surf, hidx, hits[:, 2])
            else:
                np.fmin.at(surf, hidx, hits[:, 2])
        keep = ~np.isnan(surf)
        got = surf[keep]
        expect = expect_grid[ii, jj][keep]
        if len(got):
            dev = np.abs(got - expect)
            p50, p95, mx = (float(np.percentile(dev, 50)),
                            float(np.percentile(dev, 95)),
                            float(dev.max()))
            ref = (f"(H = {wf.H:.2f})" if use_top
                   else "(expected bottom = bed plane)")
            print(f"Self-check [{what}] over {len(got)} columns: "
                  f"|got - predicted| p50 {p50:.3f}  "
                  f"p95 {p95:.3f}  max {mx:.3f} mm  {ref}")
            if p95 > 0.30:
                raise SystemExit(
                    f"ERROR: {what} check FAILED (p95 > 0.30 mm) — "
                    "non-flat warp bug; refusing to export. "
                    "Try a finer --grid_step in stage 1 or report the "
                    "model.")
            if p95 > 0.10:
                print("WARNING: flatness is marginal; consider a finer "
                      "stage-1 --grid_step.")
    else:
        print("NOTE: no conformable/anchor cells — output will be "
              "planar-ish.")

    wpath = os.path.join(outdir, f"{stem}_warped.stl")
    warped.export(wpath)
    print(f"Warped mesh -> {wpath} (watertight={warped.is_watertight}, "
          f"height {warped.bounds[1, 2]:.2f} mm)")
    print("\nNow slice it in the PrusaSlicer GUI (MK4):")
    print("  * do NOT move / rotate / scale the object on the plate")
    print("  * relative E ON, arc fitting OFF, wipe tower OFF")
    if wf.mode == "shear":
        print("  * SUPPORTS OFF — shear mode: the warped part sits "
              "flat, stage 3\n    splices its own NP supports for the "
              "raised underside")
    else:
        print("  * supports/infill/material: anything you like")
    print(f"Then: python 3_unwarp_gcode.py --field {args.field} "
          f"--gcode <exported.gcode>")


if __name__ == "__main__":
    main()
