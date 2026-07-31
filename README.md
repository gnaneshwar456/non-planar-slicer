# NP-Slicer

Non-planar (curved-layer) G-code generator for **conventional 3-axis FDM printers**
(Prusa MK4, Bambu P1S). Instead of building a part in flat horizontal layers,
NP-Slicer warps the model so a normal planar slicer lays down *curved* layers that
follow the part's geometry — improving surface finish and, for swept parts, aligning
the deposited material with the part's shape through its full thickness.

It works as a wrapper around your existing slicer (PrusaSlicer / Bambu Studio). You
never leave the slicer you already trust: NP-Slicer warps the STL, you slice it flat,
and NP-Slicer un-warps the resulting G-code back onto the curved surface — with
collision, floor, and nozzle-clearance safety checks on the final toolpaths.

> Developed for IRP SuSe 2026, Group A4. MIT licensed (see `LICENSE`).

## How it works

```
STL ─▶ 1 build field ─▶ 2 warp STL ─▶ [slice flat in your GUI] ─▶ 3 unwarp G-code ─▶ 4 verify
```

1. **Build a warp field** from the STL and printer profile (auto-picks the warp mode).
2. **Warp the STL** so its features map to a flat build volume.
3. *(manual)* **Slice the warped STL flat** in PrusaSlicer or Bambu Studio.
4. **Un-warp the G-code** — maps every move back onto the curved surface.
5. **Verify** the final G-code passes floor / burial / cone / lateral-gouge checks.

### Two warp modes (auto-selected)

- **shear** — layers follow the part's bottom-height curve through the whole
  thickness. Best for constant-thickness swept parts (e.g. tensile dogbones). The
  underside supports are generated and spliced into the G-code automatically, so
  **slicer supports must be OFF** in this mode.
- **top** — top-surface displacement decaying to a flat bottom; supports come from
  your slicer as usual.

`auto` picks **shear** when a part is sweep-like (≥70% of the footprint at constant
thickness and >1 mm bottom rise), otherwise **top**. Parts where warping gives no
real benefit degrade to a plain planar pass-through.

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Quick start (web UI, recommended)

```bash
python npslicer/server.py        # run from the repo root
```

Then open **http://localhost:5003**. The page walks you through every stage, shows
the chosen warp mode, and has the slicing instructions inline.

## Quick start (command line)

The bundled `inputs/stls/Non_Planar_Sample.stl` is a tensile dogbone — the shear-mode
demo. Run from the repo root:

```bash
# 1. Build the warp field (auto-selects shear mode for the dogbone)
python npslicer/1_build_field.py --stl inputs/stls/Non_Planar_Sample.stl --printer prusa_mk4
#    -> outputs/Non_Planar_Sample_field.npz (+ _field.png preview)

# 2. Warp the STL
python npslicer/2_warp_stl.py --field outputs/Non_Planar_Sample_field.npz
#    -> outputs/Non_Planar_Sample_warped.stl

# 3. MANUAL: slice the warped STL flat in PrusaSlicer (MK4 profile).
#    Do NOT move/rotate/scale the object. Set:
#      - "Use relative E distances"  ON
#      - "Arc fitting"               OFF
#      - Wipe tower                  OFF
#      - Supports                    OFF   (shear mode splices its own)
#    Export G-code.

# 4. Un-warp the exported G-code onto the curved surface
python npslicer/3_unwarp_gcode.py --field outputs/Non_Planar_Sample_field.npz \
       --gcode <your_exported.gcode>
#    -> outputs/Non_Planar_Sample_final.gcode

# 5. Verify the final toolpaths are collision-free
python npslicer/4_verify_gcode.py --gcode outputs/Non_Planar_Sample_final.gcode \
       --field outputs/Non_Planar_Sample_field.npz --printer prusa_mk4
#    -> want: VERIFICATION PASSED
```

Print `outputs/Non_Planar_Sample_final.gcode`.

### Trying it without a slicer installed

Step 3 is normally a manual GUI step. For testing the pipeline only, a crude planar
slicer is bundled that stands in for the GUI:

```bash
python npslicer/dev/test_slicer.py --stl outputs/Non_Planar_Sample_warped.stl \
       --out outputs/dev.gcode --layer 0.5
```

**This is for pipeline testing only — never print its output.** Real prints must go
through PrusaSlicer / Bambu Studio.

## Printer profiles

Two profiles ship in `npslicer/profiles/`:

- `prusa_mk4` — default.
- `bambu_p1s` — measured stock flat nozzle (cone 40°, tip r 0.95 mm).

G-code flavor is auto-detected at the unwarp and verify stages (Bambu Studio /
OrcaSlicer vs. PrusaSlicer). For the P1S, slice in Bambu Studio with
**Timelapse → None** and **Arc fitting → Off**, then run stage 3 with
`--printer bambu_p1s`.

> **Note:** some profile values (nozzle cone, tip-land radius, clearance/shroud
> geometry) are conservative estimates. Measure your toolhead and edit the profile
> JSON before trusting the slope caps on real hardware.

## Layout

```
npslicer/
  1_build_field.py     # stage 1 — warp field + mode selection
  2_warp_stl.py        # stage 2 — warp the STL
  3_unwarp_gcode.py    # stage 3 — un-warp the sliced G-code
  4_verify_gcode.py    # stage 4 — collision / clearance verifier
  server.py            # web UI backend  (http://localhost:5003)
  webpage.html         # web UI frontend
  common.py            # shared geometry / field math
  profiles/            # printer profiles (prusa_mk4, bambu_p1s)
  dev/test_slicer.py   # test-only planar slicer (do not print its output)
  VALIDATION.md        # validation matrix + notes
inputs/stls/           # sample part (dogbone)
outputs/               # generated fields, warped STLs, final G-code
```

See `npslicer/VALIDATION.md` for the validation results. The headline: the dogbone
prints its full 7 mm shear rise conformally through the whole thickness, with
supports spliced automatically and extrusion 100% volume-true.
