"""
NP-Slicer — shared core
==========================
Geometry-general non-planar slicing for 3-axis FDM, built around an
off-the-shelf slicer (PrusaSlicer, Prusa MK4):

    build field  ->  warp STL  ->  [slice planar in PrusaSlicer GUI]
                 ->  unwarp G-code  ->  verify

There are TWO warp modes:

  * "shear" (ShearField)  z_real = z_w + S(x, y), S = the part's
    BOTTOM height field, slope-limited. Layers follow the S-curve
    through the WHOLE thickness — the right mode for constant-
    thickness swept parts (the task dogbone / tensile specimens,
    stress-aligned deposition). Pure shear preserves layer thickness
    exactly; raised undersides get spliced grid supports (stage 3).
  * "top" (WarpField)     top-surface displacement:

    real z = z_w - f(z_w) * D(x, y)
    f(z_w) = clip(1 - (H - z_w) / d_blend, 0, 1)

  D(x, y) = H - T~(x, y)  where T~ is the part's top surface,
  gradient-limited to the nozzle-safe slope. H = max(T~).
  (There is no GLOBAL clearance clamp on D — a WINDOWED clearance check
  in the builder handles it: rise within the printhead footprint radius,
  not total rise, is what physically matters.)

Properties (all by construction):
  * top layers follow the printable top surface (conformal);
  * layers below (H - d_blend) are exactly planar -> first layers,
    supports and start G-code untouched;
  * real z <= planar z everywhere -> planar travels can never dive
    into printed material at the destination height;
  * the map is monotone in z (enforced D < d_blend), so PrusaSlicer's
    planar ordering survives the inverse warp;
  * a flat-top part gives D == 0 -> byte-equivalent planar passthrough.
"""

import json
import math
import os

import numpy as np

LAYER_MARK = ";LAYER_CHANGE"          # PrusaSlicer layer boundary
Z_MARK_RE = r"^;Z:(-?\d*\.?\d+)"      # PrusaSlicer per-layer Z comment

# G-code flavor markers. PrusaSlicer is the default path;
# Bambu Studio / OrcaSlicer (P1S) use different layer/Z markers and have
# no config footer, so the model body ends at MACHINE_END_GCODE_START.
GCODE_FLAVORS = {
    "prusa": {
        "layer_mark": ";LAYER_CHANGE",
        "z_mark_re": r"^;Z:(-?\d*\.?\d+)",
        "body_end": "; prusa_slicer_config = begin",
    },
    "bambu": {
        "layer_mark": "; CHANGE_LAYER",
        "z_mark_re": r"^; Z_HEIGHT:\s*(-?\d*\.?\d+)",
        "body_end": "; MACHINE_END_GCODE_START",
    },
}


def detect_gcode_flavor(lines):
    """Return (name, markers). Bambu Studio / OrcaSlicer emit
    '; CHANGE_LAYER'; PrusaSlicer emits ';LAYER_CHANGE'. Defaults to
    prusa when neither is conclusive, so the existing Prusa MK4 path is
    byte-for-byte unchanged."""
    has_bambu = any(ln.startswith("; CHANGE_LAYER") for ln in lines)
    has_prusa = any(ln.startswith(";LAYER_CHANGE") for ln in lines)
    if has_bambu and not has_prusa:
        return "bambu", GCODE_FLAVORS["bambu"]
    return "prusa", GCODE_FLAVORS["prusa"]

PROFILE_REQUIRED = [
    "name", "bed_x", "bed_y", "nozzle_diameter", "filament_diameter",
    "nozzle_cone_angle_deg", "tip_land_radius", "gouge_tol_mm",
    "nonplanar_clearance_h", "head_clearance_h", "head_radius",
]


def profiles_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "profiles")


def load_printer_profile(name_or_path):
    """Profile by path or by name in npslicer/profiles/. All collision-model
    values are plain JSON numbers the user can edit by hand."""
    path = name_or_path
    if not os.path.isfile(path):
        cand = os.path.join(profiles_dir(), f"{name_or_path}.json")
        if os.path.isfile(cand):
            path = cand
        else:
            avail = (sorted(f[:-5] for f in os.listdir(profiles_dir())
                            if f.endswith(".json"))
                     if os.path.isdir(profiles_dir()) else [])
            raise SystemExit(
                f"ERROR: printer profile {name_or_path!r} not found. "
                f"Available: {avail or 'none'}")
    with open(path) as f:
        prof = json.load(f)
    missing = [k for k in PROFILE_REQUIRED if k not in prof]
    if missing:
        raise SystemExit(f"ERROR: profile {path!r} missing keys: {missing}")
    if not (0 < prof["nozzle_cone_angle_deg"] < 90):
        raise SystemExit("ERROR: nozzle_cone_angle_deg must be in (0, 90)")
    prof["_path"] = path
    return prof


def load_mesh(stl_path):
    """Load an STL and repair what trimesh can repair; fail with a clear
    message otherwise. Never assumes anything about the geometry."""
    import trimesh
    if not stl_path or not os.path.exists(stl_path):
        raise SystemExit(f"ERROR: input STL not found: {stl_path!r}")
    try:
        mesh = trimesh.load(stl_path, force="mesh")
    except Exception as e:
        raise SystemExit(f"ERROR: failed to parse STL {stl_path!r}: {e}")
    if mesh is None or len(mesh.faces) == 0:
        raise SystemExit(f"ERROR: STL {stl_path!r} contains no triangles.")
    if not np.isfinite(mesh.vertices).all():
        raise SystemExit(f"ERROR: STL {stl_path!r} has non-finite vertices.")
    if not mesh.is_watertight:
        mesh.process(validate=True)
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        state = "repaired OK" if mesh.is_watertight else "STILL LEAKY"
        print(f"NOTE: input mesh was not watertight -> {state}. "
              f"Ray-cast height fields tolerate small leaks, but check "
              f"the preview image.")
    ext = mesh.bounds[1] - mesh.bounds[0]
    if min(ext[0], ext[1]) <= 0 or ext[2] <= 0:
        raise SystemExit(f"ERROR: STL {stl_path!r} degenerate extent {ext}.")
    return mesh


class HeightField:
    """Bilinear height field Z(x, y); Z_grid[i, j] = Z at (xs[j], ys[i]).
    Bilinear on purpose: the gradient-limited field is piecewise linear
    and spline interpolation overshoots at its kinks (a known interpolation failure)."""

    def __init__(self, xs, ys, Z_grid):
        from scipy.interpolate import RegularGridInterpolator
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        Z = np.asarray(Z_grid, float)
        if Z.shape != (len(ys), len(xs)):
            raise ValueError(f"Z_grid shape {Z.shape} != "
                             f"(len(ys), len(xs)) = ({len(ys)}, {len(xs)})")
        self.xs, self.ys, self.Z = xs, ys, Z
        self._ip = RegularGridInterpolator(
            (ys, xs), Z, method="linear", bounds_error=False, fill_value=None)

    def z(self, x, y):
        xc = np.clip(np.asarray(x, float), self.xs[0], self.xs[-1])
        yc = np.clip(np.asarray(y, float), self.ys[0], self.ys[-1])
        out = self._ip(np.stack([yc, xc], axis=-1))
        return float(out) if np.ndim(x) == 0 else out


def gradient_limit(F, xs, ys, tan_max, iters=None):
    """Largest field <= F whose gradient magnitude <= tan_max
    (8-neighbour cone erosion to convergence). Proven."""
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    G = F.copy()
    steps = [(0, 1, dx), (0, -1, dx), (1, 0, dy), (-1, 0, dy),
             (1, 1, math.hypot(dx, dy)), (1, -1, math.hypot(dx, dy)),
             (-1, 1, math.hypot(dx, dy)), (-1, -1, math.hypot(dx, dy))]
    if iters is None:
        iters = max(200, F.shape[0] + F.shape[1])
    for _ in range(iters):
        changed = False
        for di, dj, dist in steps:
            shifted = np.roll(G, (di, dj), axis=(0, 1))
            if di > 0:
                shifted[:di, :] = np.inf
            elif di < 0:
                shifted[di:, :] = np.inf
            if dj > 0:
                shifted[:, :dj] = np.inf
            elif dj < 0:
                shifted[:, dj:] = np.inf
            cap = shifted + tan_max * dist
            mask = G > cap
            if mask.any():
                G[mask] = cap[mask]
                changed = True
        if not changed:
            break
    return G


def flat_extend(F, mask, smooth_iters=300, freeze_cells=2):
    """Continue inside-`mask` values FLAT across the outside (nearest
    inside value + outside-only Jacobi smoothing, first rings frozen).
    Silhouette fix: the kink sits fully outside the part, so the
    field ON every silhouette equals the local surface -> no scalloped
    rims."""
    import scipy.ndimage as ndi
    dist, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
    G = np.where(mask, F, F[iy, ix])
    fixed = dist <= freeze_cells
    for _ in range(smooth_iters):
        avg = (np.roll(G, 1, 0) + np.roll(G, -1, 0) +
               np.roll(G, 1, 1) + np.roll(G, -1, 1)) / 4.0
        G = np.where(fixed, G, avg)
    return G


def bounded_extend(F, mask, xs, ys, tan_max, iters=400):
    """flat_extend + OUTSIDE-only gradient projection.

    flat_extend alone freezes the first outside rings at the value of
    the EDT-nearest inside node. At mask corners (and wherever vertical
    ray-casting speckles the mask edge) two ADJACENT outside nodes can
    inherit donors from different rim points, creating a value jump the
    erosion never sees — measured 33 deg on wave.stl with a 16.7 deg
    cap, printed as a >cap bead slope at the silhouette. This projects
    every OUTSIDE node into the
    +-tan_max*dist band of all 8 neighbours (Jacobi), with the INSIDE
    kept exactly Dirichlet, so the silhouette values stay true to the
    surface while the exterior becomes slope-consistent."""
    G = flat_extend(F, mask)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    steps = [(0, 1, dx), (0, -1, dx), (1, 0, dy), (-1, 0, dy),
             (1, 1, math.hypot(dx, dy)), (1, -1, math.hypot(dx, dy)),
             (-1, 1, math.hypot(dx, dy)), (-1, -1, math.hypot(dx, dy))]
    outside = ~mask
    for _ in range(iters):
        lower = np.full_like(G, -np.inf)
        upper = np.full_like(G, np.inf)
        for di, dj, dist in steps:
            nb = np.roll(G, (di, dj), axis=(0, 1))
            # roll wrap-around: neutralize the wrapped border rows/cols
            if di > 0:
                nb[:di, :] = G[:di, :]
            elif di < 0:
                nb[di:, :] = G[di:, :]
            if dj > 0:
                nb[:, :dj] = G[:, :dj]
            elif dj < 0:
                nb[:, dj:] = G[:, dj:]
            np.maximum(lower, nb - tan_max * dist, out=lower)
            np.minimum(upper, nb + tan_max * dist, out=upper)
        bad_band = lower > upper
        mid = 0.5 * (lower + upper)
        Gn = np.clip(G, lower, upper)
        Gn = np.where(bad_band, mid, Gn)
        changed = outside & (np.abs(Gn - G) > 1e-9)
        if not changed.any():
            break
        G = np.where(outside, Gn, G)
    return G


def grid_slope_deg(F, xs, ys):
    """Max adjacent-node slope (deg) of a grid field — the builder's
    field-level self-check (catches issue-#1-class regressions)."""
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    gx = np.abs(np.diff(F, axis=1)) / dx
    gy = np.abs(np.diff(F, axis=0)) / dy
    return math.degrees(math.atan(max(float(gx.max()), float(gy.max()))))


class WarpField:
    """The map between warped (planar-sliced) space and real space.

    D(x, y) >= 0 : how far the planar top plane H sits above the
                   printable top surface T~.
    d_blend      : depth of the transition zone below H. Monotonicity
                   needs max(D) < d_blend; the builder enforces
                   max(D) <= MONO_RATIO * d_blend.
    """
    MONO_RATIO = 0.8
    mode = "top"

    def __init__(self, xs, ys, D, H, d_blend, meta=None):
        D = np.asarray(D, float)
        if D.min() < -1e-9:
            raise ValueError("D must be >= 0")
        if D.max() > self.MONO_RATIO * d_blend + 1e-9:
            raise ValueError(
                f"max(D)={D.max():.3f} exceeds {self.MONO_RATIO} * "
                f"d_blend={d_blend:.3f}: map would not be monotone")
        self.hf_d = HeightField(xs, ys, D)
        self.xs, self.ys, self.D = xs, ys, D
        self.H = float(H)
        self.d_blend = float(d_blend)
        self.meta = dict(meta or {})

    # --- warped -> real (used on G-code) -----------------------------
    def blend(self, z_w):
        return np.clip(1.0 - (self.H - np.asarray(z_w, float))
                       / self.d_blend, 0.0, 1.0)

    def z_real(self, x, y, z_w):
        return np.asarray(z_w, float) - self.blend(z_w) * self.hf_d.z(x, y)

    def thickness(self, x, y, z_w):
        """d(z_real)/d(z_w): local vertical layer-thickness factor."""
        z_w = np.asarray(z_w, float)
        in_blend = (z_w > self.H - self.d_blend) & (z_w <= self.H)
        return np.where(in_blend,
                        1.0 - self.hf_d.z(x, y) / self.d_blend, 1.0)

    # --- real -> warped (used on the mesh) ---------------------------
    def z_warped(self, x, y, z_real):
        z_real = np.asarray(z_real, float)
        D = self.hf_d.z(x, y)
        lo = self.H - self.d_blend            # below: identity
        hi = self.H - D                       # top surface -> H
        scale = 1.0 - D / self.d_blend        # in (0, 1], monotone
        z_lin = (z_real + D - D * self.H / self.d_blend) / scale
        out = np.where(z_real <= lo, z_real,
                       np.where(z_real <= hi, z_lin, z_real + D))
        return float(out) if np.ndim(z_real) == 0 else out

    # --- persistence --------------------------------------------------
    def save(self, path, **extra):
        np.savez_compressed(
            path, xs=self.xs, ys=self.ys, D=self.D, H=self.H,
            d_blend=self.d_blend, meta_json=json.dumps(self.meta), **extra)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: field file not found: {path!r} "
                             f"(run 1_build_field.py first)")
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta_json"]))
        wf = cls(d["xs"], d["ys"], d["D"], float(d["H"]),
                 float(d["d_blend"]), meta)
        wf.raw = d
        return wf

    def self_test_roundtrip(self, n=4000, tol=1e-6, seed=0):
        """warp(unwarp(z)) == z on random samples — catches any future
        edit that breaks the piecewise inverse (an undocumented
        'warped STL not flat' class of bug)."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.xs[0], self.xs[-1], n)
        y = rng.uniform(self.ys[0], self.ys[-1], n)
        z = rng.uniform(0.0, self.H + 5.0, n)
        err = np.abs(self.z_real(x, y, self.z_warped(x, y, z)) - z)
        # inverse only defined below the top surface + above-H branch;
        # exclude the (physically empty) gap band [T~, H)
        D = self.hf_d.z(x, y)
        valid = (z <= self.H - D + 1e-9) | (z >= self.H - 1e-9)
        worst = float(err[valid].max()) if valid.any() else 0.0
        if worst > tol:
            raise SystemExit(f"ERROR: warp round-trip self-test failed "
                             f"(worst {worst:.2e} mm > {tol:.0e})")
        return worst


class ShearField:
    """The SHEAR map.

        z_real   = z_w + S(x, y)          (used on G-code)
        z_warped = z_real - S(x, y)       (used on the mesh)

    S >= 0 is the part's BOTTOM height field, gradient-limited to the
    nozzle-safe slope and continued across the silhouette with
    bounded_extend. Properties (all validated):
      * pure shear preserves layer thickness EXACTLY -> thickness
        factor == 1, E redistribution by 2D length share only;
      * for constant-thickness swept parts the layers follow BOTH
        surfaces through the whole build (stress-aligned deposition);
      * monotone per column by construction (S independent of z);
      * real z >= planar z (opposite of the top map!) -> travels must
        HOP over the sampled real surface along their path, and raised
        undersides float in real space -> stage 3 splices grid
        supports (bed -> B - gap).
    """
    mode = "shear"

    def __init__(self, xs, ys, S, meta=None):
        S = np.asarray(S, float)
        if S.min() < -1e-6:
            raise ValueError("S must be >= 0")
        self.hf_s = HeightField(xs, ys, np.maximum(S, 0.0))
        self.xs, self.ys, self.S = xs, ys, S
        self.S_max = float(S.max())
        self.meta = dict(meta or {})

    # --- warped -> real (used on G-code) -----------------------------
    def z_real(self, x, y, z_w):
        out = np.asarray(z_w, float) + self.hf_s.z(x, y)
        return float(out) if np.ndim(z_w) == 0 else out

    def thickness(self, x, y, z_w):
        """Pure shear: d(z_real)/d(z_w) == 1 everywhere."""
        z_w = np.asarray(z_w, float)
        one = np.ones_like(z_w)
        return float(one) if np.ndim(z_w) == 0 else one

    # --- real -> warped (used on the mesh) ---------------------------
    def z_warped(self, x, y, z_real):
        out = np.asarray(z_real, float) - self.hf_s.z(x, y)
        return float(out) if np.ndim(z_real) == 0 else out

    # --- persistence --------------------------------------------------
    def save(self, path, **extra):
        np.savez_compressed(
            path, xs=self.xs, ys=self.ys, S=self.S,
            meta_json=json.dumps(self.meta), **extra)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: field file not found: {path!r} "
                             f"(run 1_build_field.py first)")
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta_json"]))
        sf = cls(d["xs"], d["ys"], d["S"], meta)
        sf.raw = d
        return sf

    def self_test_roundtrip(self, n=4000, tol=1e-6, seed=0):
        """warp(unwarp(z)) == z — trivially exact for a shear, kept so
        both modes run the same firewall before anything is saved."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.xs[0], self.xs[-1], n)
        y = rng.uniform(self.ys[0], self.ys[-1], n)
        z = rng.uniform(0.0, self.S_max + 30.0, n)
        err = np.abs(self.z_real(x, y, self.z_warped(x, y, z)) - z)
        worst = float(err.max())
        if worst > tol:
            raise SystemExit(f"ERROR: shear round-trip self-test failed "
                             f"(worst {worst:.2e} mm > {tol:.0e})")
        return worst


def load_field(path):
    """Load a stage-1 field of EITHER mode. Shear fields carry an "S"
    array, top fields carry "D" — the file itself is the dispatch."""
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: field file not found: {path!r} "
                         f"(run 1_build_field.py first)")
    d = np.load(path, allow_pickle=False)
    if "S" in d.files:
        return ShearField.load(path)
    return WarpField.load(path)


def refine_mesh(V, F, max_edge, budget=4_000_000):
    """Conforming adaptive refinement: longest-edge (Rivara) bisection.

    Why not the alternatives (both documented failures):
      * trimesh.remesh.subdivide_to_size — leaves T-junction leaks
        (~3880 broken edges measured on a clean source mesh);
      * uniform subdivide loops — quadruple EVERY face per pass, so a
        mesh mixing 38 mm and 0.5 mm edges (Prop.stl) explodes to 23M
        faces and OOMs before reaching the target edge length.
    Rivara bisection splits only triangles that need it and keeps the
    mesh conforming (watertight) by construction: an edge is only ever
    split as the LONGEST edge of both its triangles, recursing into the
    neighbour first when needed.
    """
    V = [tuple(v) for v in np.asarray(V, float)]
    F = [list(f) for f in np.asarray(F, int)]

    def elen2(a, b):
        va, vb = V[a], V[b]
        return ((va[0] - vb[0]) ** 2 + (va[1] - vb[1]) ** 2
                + (va[2] - vb[2]) ** 2)

    def longest_edge(f):
        tri = F[f]
        best, bl = 0, -1.0
        for k in range(3):
            l2 = elen2(tri[k], tri[(k + 1) % 3])
            if l2 > bl:
                bl, best = l2, k
        return best, bl

    # edge (min,max) -> list of face ids
    edge_map = {}

    def reg(f):
        tri = F[f]
        for k in range(3):
            e = (min(tri[k], tri[(k + 1) % 3]),
                 max(tri[k], tri[(k + 1) % 3]))
            edge_map.setdefault(e, []).append(f)

    def unreg(f):
        tri = F[f]
        for k in range(3):
            e = (min(tri[k], tri[(k + 1) % 3]),
                 max(tri[k], tri[(k + 1) % 3]))
            lst = edge_map.get(e)
            if lst and f in lst:
                lst.remove(f)

    for f in range(len(F)):
        reg(f)
    import heapq
    t2 = max_edge * max_edge
    heap = []
    for f in range(len(F)):
        _, l2 = longest_edge(f)
        if l2 > t2:
            heapq.heappush(heap, (-l2, f, tuple(F[f])))

    def bisect_pair(f):
        """Split face f across its longest edge; recurse into the
        neighbour until the edge is the neighbour's longest too."""
        stack = [f]
        guard = 0
        while stack:
            guard += 1
            if guard > 10000:
                raise RuntimeError("rivara: propagation runaway")
            g = stack[-1]
            k, _ = longest_edge(g)
            tri = F[g]
            a, b = tri[k], tri[(k + 1) % 3]
            e = (min(a, b), max(a, b))
            nbrs = [q for q in edge_map.get(e, []) if q != g]
            if nbrs:
                q = nbrs[0]
                kq, _ = longest_edge(q)
                tq = F[q]
                eq = (min(tq[kq], tq[(kq + 1) % 3]),
                      max(tq[kq], tq[(kq + 1) % 3]))
                if eq != e:
                    stack.append(q)
                    continue
            # split edge e at midpoint in g (and matching neighbour)
            va, vb = V[e[0]], V[e[1]]
            V.append(((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2,
                      (va[2] + vb[2]) / 2))
            mid = len(V) - 1
            for t in ([g] + ([nbrs[0]] if nbrs else [])):
                tri_t = F[t]
                kt = next(kk for kk in range(3)
                          if {tri_t[kk], tri_t[(kk + 1) % 3]} ==
                          {e[0], e[1]})
                c = tri_t[(kt + 2) % 3]
                a_t, b_t = tri_t[kt], tri_t[(kt + 1) % 3]
                unreg(t)
                F[t] = [a_t, mid, c]
                reg(t)
                F.append([mid, b_t, c])
                reg(len(F) - 1)
                for nf in (t, len(F) - 1):
                    _, l2n = longest_edge(nf)
                    if l2n > t2:
                        heapq.heappush(heap, (-l2n, nf, tuple(F[nf])))
            stack.pop()

    while heap:
        if len(F) > budget:
            print(f"WARNING: refinement budget {budget} faces reached; "
                  f"largest remaining edge "
                  f"{math.sqrt(-heap[0][0]):.2f} mm")
            break
        _, f, snapshot = heapq.heappop(heap)
        if f >= len(F) or tuple(F[f]) != snapshot:
            continue                       # stale heap entry
        _, l2 = longest_edge(f)
        if l2 <= t2:
            continue
        bisect_pair(f)
    return np.asarray(V, float), np.asarray(F, int)


def lateral_slope_cap_deg(tip_r, line_w, cone_deg, gouge_tol):
    """flat-vs-pointed tip lateral cap:
    flat tip (tip_r >= line width): slope <= atan(gouge_tol / tip_r);
    pointed tip: slope <= atan((1 - tip_r/line_w) * tan(cone))."""
    if tip_r >= line_w:
        return math.degrees(math.atan(gouge_tol / tip_r)), \
            f"flat tip: atan({gouge_tol:.2f}/{tip_r:.2f})"
    frac = max(1.0 - tip_r / line_w, 0.02)
    return math.degrees(math.atan(
        frac * math.tan(math.radians(cone_deg)))), \
        f"pointed tip vs {line_w:.2f}mm bead"


def field_path(outdir, stl_path):
    stem = os.path.splitext(os.path.basename(stl_path))[0]
    stem = stem.replace(" ", "_")
    return os.path.join(outdir, f"{stem}_field.npz"), stem
