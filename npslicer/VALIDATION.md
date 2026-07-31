# Validation

Sandbox run using the bundled dev planar slicer (`dev/test_slicer.py`) as a
stand-in for the PrusaSlicer GUI step.

Profile: `prusa_mk4` (SPECULATIVE Nextruder values, including
`shroud_radius` = 8 mm — **measure and edit these before trusting them on
hardware**). Effective slope cap θ = 16.7° (lateral flat-tip cap), grid
erosion at 12.0° (θ/√2). Windowed clearance check: rise within shroud
r = 8 mm = 1.70 ≤ 4.0 mm, within head r = 35 mm = 7.42 ≤ 10.0 mm → no
field-slope clamp needed.

Every part ran the full chain: build field → warp → planar slice (dev
slicer, coarse 0.4–0.6 mm layers to fit the sandbox CPU budget) → unwarp →
verify. "PASS" = all four verifier checks (floor / burial / cone+housing /
lateral tip-gouge) clean.

> Only `Non_Planar_Sample.stl` (the dogbone) ships in `inputs/stls/`. The
> other rows are additional test parts used to exercise mode selection and
> the degenerate cases; bring your own STLs to reproduce them.

| STL               | Mode (auto)        | Field max (mm) | Conformal        | Warp-flatness p95      | Verify |
|-------------------|--------------------|---------------:|------------------|-----------------------:|--------|
| Non_Planar_Sample | **SHEAR**          | S_max 7.00     | S==B on 87%      | 0.000 (bottom-on-bed)  | PASS   |
| non-planar-small  | top (see note 1)   | D 1.93         | 33% top          | 0.001                  | PASS   |
| wave              | top                | D 2.07         | 46% top          | 0.005                  | PASS   |
| Simple block      | top                | D 3.40         | 100% top         | 0.015                  | PASS   |
| Prop              | top                | D 3.39         | 54% top (blades) | 0.011                  | PASS   |
| simple_sphere     | top                | D 6.08         | 9% (top cap)     | 0.004                  | PASS   |
| slant             | top → PASSTHROUGH  | 0 (benefit 0%) | (planar)         | n/a                    | PASS   |
| simple_cube       | PASSTHROUGH        | 0.00           | (flat top)       | n/a                    | PASS   |

Headline result: **the dogbone prints its full 7 mm shear rise conformally
through the whole thickness**, with 33 spliced NP support levels and
extrusion 100.0% volume-true (pure shear, no thinning term).

Notes:

1. `non-planar-small`: its underside is FLAT on the bed (max B rise
   0.00 mm), so it is not sweep-like and shear would be a no-op (S ≡ 0).
   Auto correctly picks top mode.
2. `slant`: auto passthrough by the near-zero-benefit rule. 0.0% of the
   footprint is conformable AND displaced > 0.3 mm. The 35–45° face
   exceeds any realistic slope cap — stairs there are 3-axis physics
   (documented, not fought). `--mode top` forces the warp.
3. The benefit rule is implemented as "conformable AND displaced
   > 0.3 mm fraction < 5%" (`--min_benefit`), not a raw "conformable
   < 10%": the raw metric would wrongly flip the sphere (9% cap, all
   genuinely displaced → benefit 7.6%, stays non-planar) and would NOT
   flip slant (27% raw, all undisplaced).
4. `simple_sphere`: windowed clearance lets the printable apex cap relieve
   deeper (D_max 6.08); same 9% conformal cap area; PASS.
5. `Prop`: the monotone-map limit is clamped per column (graceful
   degradation) rather than scaling the whole field down, giving 54%
   conformable at D_max 3.39.
6. Shear-mode support pacing: after a spliced support block, the next
   travel's hop must be planned from the real nozzle XY (`rx, ry` in the
   unwarp stage), not the stale input position — otherwise the hop dives
   through the fresh towers (caught by the verifier's [burial]/[cone]
   checks).
7. Forced-mode sanity: `--mode shear` on a flat-bottom part reports PLANAR
   PASSTHROUGH (S ≡ 0); `--mode top` on the dogbone yields the full
   D_max 7.00 with 88% conformable top.
8. The dev slicer output is for pipeline testing only — real prints go
   through the PrusaSlicer GUI per the README (shear mode: supports OFF).
