using ElectronDensityAnalysis: SphericalScalarMesh, great_circle_distance
using Serialization: deserialize
using StaticArrays
using LinearAlgebra
using Statistics
using PlotlyJS
using NearestNeighbors, Optim

@enum SimplexType Vertex=0 Edge=1 Triangle=2

struct DiscreteGradient
    # Pairings: maps simplex index to paired simplex index. 0 if unpaired.
    v_to_e::Vector{Int}
    e_to_v::Vector{Int}
    e_to_t::Vector{Int}
    t_to_e::Vector{Int}

    # Critical Cells (Indices into mesh.positions, mesh.edges, mesh.triangles)
    critical_vertices::Vector{Int}  # Minima
    critical_edges::Vector{Int}     # Saddles
    critical_triangles::Vector{Int} # Maxima
end

mutable struct ChemicalTopology
    # Critical Points (Cartesian Coordinates on Sphere)
    maxima::Vector{SVector{3,Float64}}
    minima::Vector{SVector{3,Float64}}
    saddles::Vector{SVector{3,Float64}}

    # Saddle connectivity: ridge_edges[k] = (max_i, max_j), valley_edges[k] = (min_a, min_b)
    # Index-aligned with saddles. SVector(0,0) marks fallback saddles without a valley arc.
    ridge_edges::Vector{SVector{2,Int}}
    valley_edges::Vector{SVector{2,Int}}

    # The 1-Skeleton (Optimized Splines)
    # Stored as vectors of control points or high-res sample points
    # Key = (Type, Index1, Index2)
    connectors::Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}
end

struct FieldSampler
    mesh::SphericalScalarMesh
    tree::KDTree
    node_to_triangles::Vector{Vector{Int}} # Map: NodeIdx -> [TriIdx...]
end

function build_sampler(mesh::SphericalScalarMesh)
    # 1. Build KDTree for fast nearest-neighbor lookup
    # Convert SVectors to matrix for KDTree
    data = reinterpret(reshape, Float64, mesh.positions)
    tree = KDTree(data)

    # 2. Build Adjacency Map (Node -> Incident Triangles)
    node_to_triangles = [Int[] for _ in 1:length(mesh.positions)]
    for (t_idx, tri) in enumerate(mesh.triangles)
        push!(node_to_triangles[tri[1]], t_idx)
        push!(node_to_triangles[tri[2]], t_idx)
        push!(node_to_triangles[tri[3]], t_idx)
    end

    return FieldSampler(mesh, tree, node_to_triangles)
end

"""
    sample_field(sampler, r)

Evaluates the scalar field at position `r` (SVector{3}) using barycentric interpolation.
"""
function sample_field(sampler::FieldSampler, r::SVector{3,Float64})
    # 1. Find nearest mesh node
    idxs, dists = knn(sampler.tree, r, 1)
    nearest_node = idxs[1]

    # 2. Check incident triangles to find the enclosure
    # Project r onto the plane of each triangle to find barycentric coords
    best_val = sampler.mesh.values[nearest_node] # Fallback
    min_dist_outside = Inf

    for t_idx in sampler.node_to_triangles[nearest_node]
        tri = sampler.mesh.triangles[t_idx]
        p1 = sampler.mesh.positions[tri[1]]
        p2 = sampler.mesh.positions[tri[2]]
        p3 = sampler.mesh.positions[tri[3]]

        # Barycentric calculation (Triangle is 3D, Point is 3D)
        # Solve: r = u*p1 + v*p2 + w*p3, subject to u+v+w=1
        # Use vector area method (robust)

        area_total = norm(cross(p2 - p1, p3 - p1))

        # Sub-areas
        # Note: We must ensure r is on the same plane or projected. 
        # For sphere, we usually approximate flat triangle.

        # Vector areas
        na = cross(p2 - p1, p3 - p1) # Normal of triangle

        # Check if r is roughly aligned with normal (simple front-face check)
        # if dot(r, na) < 0; continue; end 

        # Barycentric coordinates via projection
        # u = Area(r, p2, p3) / AreaTotal
        # v = Area(r, p3, p1) / AreaTotal
        # w = Area(r, p1, p2) / AreaTotal

        # Use cross product magnitudes
        u = norm(cross(p2 - r, p3 - r)) / area_total
        v = norm(cross(p3 - r, p1 - r)) / area_total
        w = norm(cross(p1 - r, p2 - r)) / area_total

        # Check if inside triangle (sum ≈ 1)
        # Since r is on sphere and tri is flat chord, sum will be > 1.
        # But u,v,w ratios should be correct.
        sum_uvw = u + v + w

        # Normalize weights
        u /= sum_uvw
        v /= sum_uvw
        w /= sum_uvw

        # Rough check: if point projects inside triangle, coefficients are all positive
        # However, due to curvature, simple cross-product areas are always positive.
        # We need a proper "inside" check. 
        # For now, relying on KNN is usually accurate enough to pick the right patch.
        # Let's just weighted-average the triangle that is "closest" to normal?

        # Improvement: Just interpolate on the nearest triangle found. 
        # Since we picked the nearest node, we are likely in the fan.
        vals = (
            sampler.mesh.values[tri[1]],
            sampler.mesh.values[tri[2]],
            sampler.mesh.values[tri[3]],
        )
        val_interp = u*vals[1] + v*vals[2] + w*vals[3]
        return val_interp
    end

    return best_val
end

"""
    relax_minima_constrained(pred_minima, sampler, origin, radius; stiffness=10.0)

Relaxes predicted minima coordinates to minimize Energy = f(r) + k*(dist_to_pred)^2.
Correctly handles spheres not centered at (0,0,0).
"""
function relax_minima_constrained(
    pred_minima::Vector{SVector{3,Float64}},
    sampler::FieldSampler,
    origin::SVector{3,Float64},
    radius::Float64;
    stiffness=100.0,
)
    refined_minima = SVector{3,Float64}[]

    for (i, p_init) in enumerate(pred_minima)

        # Objective Function
        function objective(x)
            r_curr = SVector{3}(x)

            # 1. Project x to the SURFACE of the sphere (Relative to Origin)
            # Vector from sphere center to point
            v = r_curr - origin

            # Re-normalize to exact radius and shift back
            r_proj = origin + normalize(v) * radius

            # 2. Sample field at the projected world coordinate
            val = sample_field(sampler, r_proj)

            # 3. Elastic Penalty (Distance between surface point and prediction)
            dist_sq = great_circle_distance(r_proj, p_init, origin)^2

            return val + stiffness * dist_sq
        end

        # Start optimization at the predicted world coordinate
        res = optimize(
            objective, Vector(p_init), NelderMead(), Optim.Options(; iterations=50)
        )

        # Final Projection to ensure exact radius
        v_final = SVector{3}(Optim.minimizer(res)) - origin
        p_final = origin + normalize(v_final) * radius

        push!(refined_minima, p_final)
    end

    return refined_minima
end

"""
    relax_saddles_geodesic(maxima, delaunay_edges, sampler, origin, radius; stiffness=100.0, max_iters=10)

Relaxes saddles using an alternating minimax coordinate descent. 
It iteratively minimizes along the Ridge (Max-Max direction) and maximizes along 
the orthogonal Valley, anchored by an elastic penalty to the geometric prediction.
"""
function relax_saddles_geodesic(
    maxima::Vector{SVector{3,Float64}},
    delaunay_edges::Vector{SVector{2,Int}},
    sampler::FieldSampler,
    origin::SVector{3,Float64},
    radius::Float64;
    stiffness=100.0,
    max_iters=10,
)
    refined_saddles = SVector{3,Float64}[]

    # Search bounds for the 1D line searches (in radians)
    # ~0.15 radians is about 8.5 degrees, a safe local search window
    search_window = 0.15

    for edge in delaunay_edges
        p1 = maxima[edge[1]]
        p2 = maxima[edge[2]]

        v1 = normalize(p1 - origin)
        v2 = normalize(p2 - origin)
        omega = acos(clamp(dot(v1, v2), -1.0, 1.0))

        if abs(omega) < 1e-6
            push!(refined_saddles, p1)
            continue
        end

        # Anchor point (Voronoi prediction midpoint)
        so = sin(omega)
        mid_scale = sin(0.5 * omega) / so
        v_mid = mid_scale * v1 + mid_scale * v2
        r_mid = origin + normalize(v_mid) * radius

        r_curr = r_mid # Start guess at geometric center

        # The chord vector gives us the general "Ridge" direction
        chord_dir = normalize(p2 - p1)

        # Alternating Minimax Loop
        for iter in 1:max_iters
            r_old = r_curr
            normal = normalize(r_curr - origin)

            # --- STEP 1: MINIMIZE ALONG RIDGE ---
            # Project chord direction onto the local tangent plane
            u_dir = normalize(chord_dir - dot(chord_dir, normal) * normal)

            function ridge_obj(t)
                # Move along tangent and project back to sphere surface
                r_test = origin + normalize(normal + t * u_dir) * radius
                val = sample_field(sampler, r_test)
                dist_sq = great_circle_distance(r_test, r_mid, origin)^2

                # Minimize Value, Penalize Distance
                return val + stiffness * dist_sq
            end

            # Optim's 1D search handles bounds natively
            res_u = optimize(ridge_obj, -search_window, search_window)
            r_curr = origin + normalize(normal + res_u.minimizer * u_dir) * radius

            # --- STEP 2: MAXIMIZE ALONG VALLEY ---
            normal = normalize(r_curr - origin)

            # Re-project ridge vector at the new location
            u_dir = normalize(chord_dir - dot(chord_dir, normal) * normal)

            # The valley direction is perfectly orthogonal to the ridge and normal
            v_dir = normalize(cross(normal, u_dir))

            function valley_obj(t)
                r_test = origin + normalize(normal + t * v_dir) * radius
                val = sample_field(sampler, r_test)
                dist_sq = great_circle_distance(r_test, r_mid, origin)^2

                # We want to MAXIMIZE the field value while staying close to the anchor.
                # Maximize (Value - Penalty) == Minimize (-Value + Penalty)
                return -val + stiffness * dist_sq
            end

            res_v = optimize(valley_obj, -search_window, search_window)
            r_curr = origin + normalize(normal + res_v.minimizer * v_dir) * radius

            # --- CONVERGENCE CHECK ---
            if norm(r_curr - r_old) < 1e-6
                break
            end
        end

        push!(refined_saddles, r_curr)
    end

    return refined_saddles
end

# --- 1. Spline Evaluation Helpers ---

"""
    eval_projected_spline(p1, p2, controls, t, origin, radius)

Evaluates a Projected Bezier Spline at parameter t [0,1].
Refactored to accept generic AbstractVectors to prevent MethodErrors with empty arrays.
"""
function eval_projected_spline(p1, p2, controls, t, origin, radius)

    # Shift to local coords relative to origin
    v1 = p1 - origin
    v2 = p2 - origin

    mt = 1.0 - t

    if isempty(controls)
        # Linear (Order 0)
        v = mt * v1 + t * v2
    elseif length(controls) == 1
        # Quadratic (Order 1)
        v_ctrl = controls[1] - origin
        v = (mt^2) * v1 + (2*t*mt) * v_ctrl + (t^2) * v2
    else
        # Cubic (Order 2)
        v_c1 = controls[1] - origin
        v_c2 = controls[2] - origin
        v = (mt^3) * v1 + (3*t*mt^2) * v_c1 + (3*t^2*mt) * v_c2 + (t^3) * v2
    end

    # Project to Sphere
    return origin + normalize(v) * radius
end

# --- 2. Typed Path Integral Objective ---

"""
    path_integral_objective(control, p1, p2, sampler, origin, radius, type;
                            samples, w_mono, skewness, t_center)

Evaluates the path-integral cost of a quadratic Bézier arc defined by a single optional
midpoint control. `control = nothing` gives the geodesic (no Bézier bend).

For `:ridge` and `:valley` arcs the unimodal shape penalty `w_mono` is applied:
- `:ridge` (∪): enforces descent to a single interior minimum then ascent back to p2.
- `:valley` (∩): enforces ascent to a single interior maximum then descent back to p2.
No shape penalty is applied for `:descent` slope arcs.

The `skewness` and `t_center` parameters warp the distribution of sample points along the arc:
- `skewness = 1`: uniform spacing in Bézier parameter `t` ∈ [0,1] (no warp).
- `skewness > 1`: samples cluster densely near `t_center` and sparsely near the endpoints.
                  Use to de-emphasise the terminal maxima/minima and emphasise the saddle.
- `skewness < 1`: samples cluster near the endpoints and sparsely near `t_center`.
- `t_center`:     The `t` value where the expected saddle lies on this arc. Derived from the
                  probe extremum index each optimizer iteration; defaults to `0.5`.

The warp is a piecewise power-law that preserves `t = 0`, `t = t_center`, and `t = 1` exactly:
  `s ≤ t_center : t = t_center * (s / t_center)^(1/skewness)`
  `s > t_center : t = 1 − (1−t_center) * ((1−s)/(1−t_center))^(1/skewness)`
"""
function path_integral_objective(
    control::Union{SVector{3,Float64},Nothing},
    p1,
    p2,
    sampler,
    origin,
    radius,
    type;
    samples=50,
    w_mono=1.0,
    skewness=1.0,
    t_center=0.5,
)
    # Build the controls vector: quadratic Bézier with one midpoint, or geodesic
    controls = isnothing(control) ? SVector{3,Float64}[] : [control]

    # Integration loop — collect sampled values for ∫ρ ds and shape enforcement
    total_val = 0.0
    total_len = 0.0
    path_values = Vector{Float64}(undef, samples)

    dt = 1.0 / (samples - 1)
    prev_pos = p1

    for i in 1:samples
        s = (i - 1) * dt
        # Piecewise power-law warp: cluster sample points near t_center
        t = if skewness ≈ 1.0 || t_center <= 1e-9 || t_center >= 1.0 - 1e-9
            s
        elseif s <= t_center
            t_center * (s / t_center)^(1.0 / skewness)
        else
            1.0 - (1.0 - t_center) * ((1.0 - s) / (1.0 - t_center))^(1.0 / skewness)
        end
        pos = eval_projected_spline(p1, p2, controls, t, origin, radius)
        val = sample_field(sampler, pos)

        path_values[i] = val
        step_len = norm(pos - prev_pos)
        total_len += step_len
        total_val += val * step_len
        prev_pos = pos
    end

    # Unimodal shape enforcement
    # Ridge (max→sad→max): ∪ — decreases to one interior minimum, then increases.
    # Valley (min→sad→min): ∩ — increases to one interior maximum, then decreases.
    mono_penalty = 0.0
    if type == :ridge || type == :max_saddle
        k = argmin(path_values)
        for i in 1:(k - 1)
            mono_penalty += max(0.0, path_values[i + 1] - path_values[i])^2
        end
        for i in k:(samples - 1)
            mono_penalty += max(0.0, path_values[i] - path_values[i + 1])^2
        end
    elseif type == :valley || type == :min_saddle
        k = argmax(path_values)
        for i in 1:(k - 1)
            mono_penalty += max(0.0, path_values[i] - path_values[i + 1])^2
        end
        for i in k:(samples - 1)
            mono_penalty += max(0.0, path_values[i + 1] - path_values[i])^2
        end
    end

    final_reg = w_mono * mono_penalty

    if type == :valley || type == :min_saddle
        return total_val + final_reg
    elseif type == :ridge || type == :max_saddle
        return -total_val + final_reg
    else
        return total_val + final_reg
    end
end

# --- 3. The Main Optimizer ---

"""
    optimize_spline_connector(p1, p2, mesh, sampler; type, threshold, w_proximity, w_mono,
                               probe_smooth_pct, max_iters, conv_tol)

Finds the optimal quadratic Bézier arc between `p1` and `p2` on the sphere surface using an
iterative 1-D Brent search over a single perpendicular degree of freedom.

Algorithm (per iteration):
1. Sample the current path (great arc on iteration 1; best Bézier spline on subsequent
   iterations) at `n_probe=60` evenly spaced points and evaluate the field at each.
2. Apply a discrete Gaussian kernel (width `probe_smooth_pct`) to suppress noise.
3. Find the control-point reference position `c_init` as the smoothed-field extremum:
   `argmin` for `:ridge` (∪ shape: interior minimum = saddle),
   `argmax` for `:valley` (∩ shape: interior maximum = saddle), midpoint for `:descent`.
4. Compute `perp_dir` as the local path-normal in the tangent plane of the sphere at
   `c_init`, using finite differences of the adjacent sampled path points. On iteration 1
   these are great-arc points; on subsequent iterations they lie on the bent spline, so the
   search direction evolves to match the current geometry.
5. Search a scalar displacement `δ` ∈ [−arc_len/2, +arc_len/2] via Brent's method, where
   `arc_len` is the current path's arc length (tightens as the spline converges):
       ctrl = origin + normalize((c_init − origin) + δ·perp_dir) · radius
6. Converge when `‖new_ctrl − best_ctrl‖ < conv_tol · arc_len`.

After all iterations the final control point is accepted only if the improvement over the
pure geodesic exceeds `threshold`; otherwise the geodesic is returned.

# Keyword Arguments
- `type`:             Path objective:
                      `:ridge` — maximizes ∫ρ ds (high-density ridge, max→max arcs).
                      `:valley` — minimizes ∫ρ ds (low-density valley, min→min arcs).
                      `:descent` — minimizes ∫ρ ds (slope/min→max arcs, no shape constraint).
- `threshold`:        Minimum relative improvement over the geodesic needed to accept the
                      final bent arc. Lower → more curvature; higher → geodesic-like. Default: `0.005`.
- `w_proximity`:      Weight on the δ² proximity penalty each iteration. Penalizes displacement
                      from the current reference point, biasing toward the nearest local optimum.
                      Higher → smaller per-iteration steps; lower → more freedom. Default: `0.1`.
- `w_mono`:           Weight on the unimodal shape penalty for `:ridge`/`:valley` arcs.
                      Higher → stronger ∪/∩ enforcement. Not applied for `:descent`. Default: `1.0`.
- `probe_smooth_pct`: Half-width of the Gaussian kernel (fraction of `n_probe`) for smoothing
                      field values before selecting `c_init`. Default: `0.10`.
- `max_iters`:        Maximum number of refinement iterations. Default: `5`.
- `conv_tol`:         Convergence tolerance as a fraction of the current arc length;
                      iteration stops when the control point moves less than this. Default: `1e-3`.
- `skewness`:         Exponent for the piecewise power-law sample-point warp in the path-integral
                      objective. `1.0` — uniform; `> 1.0` — denser near the arc's expected saddle
                      (derived from the probe extremum each iteration), less weight on endpoints.
                      Default: `2.0`.
"""
function optimize_spline_connector(
    p1::SVector{3,Float64},
    p2::SVector{3,Float64},
    mesh::SphericalScalarMesh,
    sampler::FieldSampler;
    type=:valley,
    threshold=0.005,
    w_proximity=0.1,
    w_mono=1.0,
    probe_smooth_pct=0.10,
    max_iters=5,
    conv_tol=1e-3,
    skewness=2.0,
)
    origin = mesh.origin
    radius = mesh.radius
    n_probe = 60

    # --- Reusable helper: 1-D discrete Gaussian smoothing ---
    hw = max(1, round(Int, probe_smooth_pct * n_probe / 2))
    σ = hw / 2.0
    function gaussian_smooth_1d(vals)
        n = length(vals)
        out = similar(vals)
        for i in 1:n
            w_sum = 0.0;
            v_sum = 0.0
            for j in max(1, i - hw):min(n, i + hw)
                w = exp(-((j - i)^2) / (2σ^2))
                v_sum += w * vals[j]
                w_sum += w
            end
            out[i] = v_sum / w_sum
        end
        return out
    end

    # --- Reusable helper: perpendicular direction at pts[k] ---
    function path_perp_dir(pts, k)
        c_pt = pts[k]
        sph_normal = normalize(c_pt - origin)
        lo = max(1, k - 1);
        hi = min(length(pts), k + 1)
        tang_raw = pts[hi] - pts[lo]
        arc_tang = if norm(tang_raw) > 1e-10
            normalize(tang_raw)
        else
            arb = if abs(sph_normal[3]) < 0.9
                SVector(0.0, 0.0, 1.0)
            else
                SVector(1.0, 0.0, 0.0)
            end
            normalize(cross(sph_normal, arb))
        end
        raw_perp = cross(sph_normal, arc_tang)
        return if norm(raw_perp) > 1e-8
            normalize(raw_perp)
        else
            arb = abs(sph_normal[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
            normalize(cross(sph_normal, arb))
        end
    end

    # --- Reusable helper: extremum index along a smoothed value vector ---
    function extremum_k(smoothed)
        if type == :ridge || type == :max_saddle
            argmin(smoothed)
        elseif type == :valley || type == :min_saddle
            argmax(smoothed)
        else
            length(smoothed) ÷ 2
        end
    end

    # --- Baseline: pure geodesic cost (for final threshold check) ---
    cost_geodesic = path_integral_objective(
        nothing,
        p1,
        p2,
        sampler,
        origin,
        radius,
        type;
        w_mono=w_mono,
        skewness=skewness,
        t_center=0.5,
    )

    best_control = nothing  # start from geodesic
    t_c = 0.5               # updated each iteration from probe extremum

    # --- Iterative refinement ---
    for iter in 1:max_iters
        # Sample the current path: great arc (iter 1) or best Bézier spline (iter 2+)
        current_controls = isnothing(best_control) ? SVector{3,Float64}[] : [best_control]
        pts = [
            eval_projected_spline(
                p1, p2, current_controls, (i - 1) / (n_probe - 1), origin, radius
            ) for i in 1:n_probe
        ]

        # Smooth field values along the current path
        raw_vals = [sample_field(sampler, p) for p in pts]
        smoothed = gaussian_smooth_1d(raw_vals)

        # Reference control-point position: extremum along smoothed current path
        k_init = extremum_k(smoothed)
        t_c = (k_init - 1) / (n_probe - 1)   # saddle t-parameter for this arc
        c_init = pts[k_init]

        # Local path-normal at c_init (uses current spline tangent from iteration 2+)
        perp_dir = path_perp_dir(pts, k_init)

        # Displacement bound from current arc length
        arc_len = sum(norm(pts[i + 1] - pts[i]) for i in 1:(n_probe - 1))
        max_disp = 0.5 * arc_len

        # 1-D Brent search
        c_init_vec = c_init - origin
        function f_1d(δ)
            ctrl = origin + normalize(c_init_vec + δ * perp_dir) * radius
            return path_integral_objective(
                ctrl,
                p1,
                p2,
                sampler,
                origin,
                radius,
                type;
                w_mono=w_mono,
                skewness=skewness,
                t_center=t_c,
            ) + w_proximity * δ^2
        end

        res = optimize(f_1d, -max_disp, max_disp, Brent())
        δ_opt = Optim.minimizer(res)
        new_ctrl = origin + normalize(c_init_vec + δ_opt * perp_dir) * radius

        # Convergence check: stop if control point barely moved
        converged =
            !isnothing(best_control) && norm(new_ctrl - best_control) < conv_tol * arc_len
        best_control = new_ctrl
        converged && break
    end

    # --- Final threshold check: accept only if genuinely better than the geodesic ---
    cost_final = path_integral_objective(
        best_control,
        p1,
        p2,
        sampler,
        origin,
        radius,
        type;
        w_mono=w_mono,
        skewness=skewness,
        t_center=t_c,
    )
    if (cost_geodesic - cost_final) / (abs(cost_geodesic) + 1e-9) <= threshold
        best_control = nothing
    end

    # --- Sample 40 output points ---
    controls_vec = isnothing(best_control) ? SVector{3,Float64}[] : [best_control]
    final_points = SVector{3,Float64}[]
    n_out = 40
    for i in 1:n_out
        t = (i - 1) / (n_out - 1)
        push!(final_points, eval_projected_spline(p1, p2, controls_vec, t, origin, radius))
    end

    return final_points
end

"""
    find_spline_intersection(path1, path2, origin, radius)

Finds the exact crossing point of two discrete paths on a sphere using three phases:
- Phase 1: Detect the crossing segment pair via signed triple products (spherical
           orientation test — the spherical analog of the 2D winding test).
- Phase 2: Analytically solve the closest-approach system for the two crossing
           line segments, then project the midpoint to the sphere surface.
- Phase 3: Fallback to closest sampled-point-pair heuristic when no sign-change
           crossing is detected (boundary / degenerate geometry).
"""
function find_spline_intersection(
    path1::Vector{SVector{3,Float64}},
    path2::Vector{SVector{3,Float64}},
    origin::SVector{3,Float64},
    radius::Float64,
)
    # --- Phase 1: Crossing detection via spherical orientation ---
    # Arc (a→b) and arc (c→d) cross on the sphere iff:
    #   sign(dot(cross(a-o, b-o), c-o)) ≠ sign(dot(cross(a-o, b-o), d-o))  AND
    #   sign(dot(cross(c-o, d-o), a-o)) ≠ sign(dot(cross(c-o, d-o), b-o))
    crossing_i = 0
    crossing_j = 0
    found = false

    @inbounds for i in 1:(length(path1) - 1)
        a = path1[i] - origin
        b = path1[i + 1] - origin
        n_ab = cross(a, b)
        for j in 1:(length(path2) - 1)
            c = path2[j] - origin
            d = path2[j + 1] - origin
            # Test 1: c and d on opposite sides of the plane through (o, a, b)
            sign_c = dot(n_ab, c)
            sign_d = dot(n_ab, d)
            if sign_c * sign_d >= 0.0
                ;
                continue;
            end
            # Test 2: a and b on opposite sides of the plane through (o, c, d)
            n_cd = cross(c, d)
            if dot(n_cd, a) * dot(n_cd, b) >= 0.0
                ;
                continue;
            end
            crossing_i = i
            crossing_j = j
            found = true
            break
        end
        if found
            ;
            break;
        end
    end

    if found
        # --- Phase 2: Analytical closest-approach for the crossing segment pair ---
        a = path1[crossing_i]
        b = path1[crossing_i + 1]
        c = path2[crossing_j]
        d = path2[crossing_j + 1]
        ab = b - a
        cd = d - c
        ac = c - a
        # Minimise ||a + s*ab - c - t*cd||² over s,t ∈ [0,1].
        # Setting the gradient to zero gives the 2×2 linear system:
        #   [ dot(ab,ab)  -dot(ab,cd) ] [s]   [ dot(ac,ab) ]
        #   [ dot(ab,cd)  -dot(cd,cd) ] [t] = [ dot(ac,cd) ]
        A11 = dot(ab, ab)
        A12 = -dot(ab, cd)
        A21 = dot(ab, cd)
        A22 = -dot(cd, cd)
        b1 = dot(ac, ab)
        b2 = dot(ac, cd)
        det = A11 * A22 - A12 * A21
        if abs(det) > 1e-14
            s = clamp((A22 * b1 - A12 * b2) / det, 0.0, 1.0)
            t = clamp((A11 * b2 - A21 * b1) / det, 0.0, 1.0)
        else
            s = 0.5
            t = 0.5
        end
        midpoint = (a + s * ab + c + t * cd) / 2.0
        return origin + normalize(midpoint - origin) * radius
    end

    # --- Phase 3: Fallback — closest sampled-point-pair heuristic ---
    min_dist_sq = Inf
    best_p = (path1[1] + path2[1]) / 2.0
    for p1 in path1, p2 in path2
        dist_sq = sum(abs2, p1 - p2)
        if dist_sq < min_dist_sq
            min_dist_sq = dist_sq
            best_p = (p1 + p2) / 2.0
        end
    end
    return origin + normalize(best_p - origin) * radius
end

"""
    optimize_ridge_valley_arcs(maxima, minima, delaunay_edges, voronoi_edges, sampler, origin, radius;
                                threshold, w_proximity, w_mono, probe_smooth_pct, max_iters, conv_tol)

For each Delaunay edge:
- If the dual Voronoi edge is known (interior edges), optimizes the full max→max ridge arc
  and the full min→min valley arc as splines, then derives the saddle as their intersection.
- If the Voronoi edge is `nothing` (boundary edge), falls back to `relax_saddles_geodesic`.

Returns:
- `saddles`          : Vector of saddle positions, 1-to-1 with input edges
- `connectors`       : Dict with (:max_max, i, j) ridge arcs and (:min_min, a, b) valley arcs
- `ridge_edges_out`  : SVector{2,Int} of max indices for each saddle
- `valley_edges_out` : SVector{2,Int} of min indices for each saddle;
                       SVector(0,0) marks fallback saddles that have no valley arc.

Spline tuning kwargs (forwarded to `optimize_spline_connector`):
- `threshold`:        Minimum relative improvement over the geodesic to accept a bent arc (default `0.005`).
- `w_proximity`:      δ² penalty weight — higher keeps control points closer to their initial
                      on-arc position (default `0.1`).
- `w_mono`:           Unimodal shape penalty weight for ridge (∪) and valley (∩) arcs (default `1.0`).
- `probe_smooth_pct`: Gaussian kernel half-width as a fraction of n_probe for initial control-point
                      placement from the smoothed field (default `0.10`).
- `max_iters`:        Maximum refinement iterations per arc (default `5`).
- `conv_tol`:         Convergence tolerance as a fraction of arc length (default `1e-3`).
- `skewness`:         Sample-point density exponent; `> 1` clusters samples near the per-arc
                      saddle (derived from probe extremum each iteration). Default: `2.0`.
"""
function optimize_ridge_valley_arcs(
    maxima::Vector{SVector{3,Float64}},
    minima::Vector{SVector{3,Float64}},
    delaunay_edges::Vector{SVector{2,Int}},
    voronoi_edges::Vector{Union{SVector{2,Int},Nothing}},
    sampler::FieldSampler,
    origin::SVector{3,Float64},
    radius::Float64;
    threshold=0.005,
    w_proximity=0.1,
    w_mono=1.0,
    probe_smooth_pct=0.10,
    max_iters=5,
    conv_tol=1e-3,
    skewness=2.0,
)
    saddles = SVector{3,Float64}[]
    connectors = Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}()
    ridge_edges_out = SVector{2,Int}[]
    valley_edges_out = SVector{2,Int}[]

    fallback_local_indices = Int[]     # positions in delaunay_edges needing fallback
    fallback_del_edges = SVector{2,Int}[]

    for k in eachindex(delaunay_edges)
        v_edge = voronoi_edges[k]
        d_edge = delaunay_edges[k]

        if isnothing(v_edge) || v_edge[1] > length(minima) || v_edge[2] > length(minima)
            push!(fallback_local_indices, k)
            push!(fallback_del_edges, d_edge)
            continue
        end

        mi, mj = d_edge[1], d_edge[2]        # indices into maxima
        va, vb = v_edge[1], v_edge[2]        # indices into pred_minima / minima

        println("  Arc $k: ridge max[$mi]↔max[$mj], valley min[$va]↔min[$vb]")

        # Full ridge arc: max_i → max_j  (maximize ∫ρ ds)
        ridge_path = optimize_spline_connector(
            maxima[mi],
            maxima[mj],
            sampler.mesh,
            sampler;
            type=:ridge,
            threshold=threshold,
            w_proximity=w_proximity,
            w_mono=w_mono,
            probe_smooth_pct=probe_smooth_pct,
            max_iters=max_iters,
            conv_tol=conv_tol,
            skewness=skewness,
        )

        # Full valley arc: min_a → min_b  (minimize ∫ρ ds)
        valley_path = optimize_spline_connector(
            minima[va],
            minima[vb],
            sampler.mesh,
            sampler;
            type=:valley,
            threshold=threshold,
            w_proximity=w_proximity,
            w_mono=w_mono,
            probe_smooth_pct=probe_smooth_pct,
            max_iters=max_iters,
            conv_tol=conv_tol,
            skewness=skewness,
        )

        # Saddle = exact intersection of the two arcs
        saddle = find_spline_intersection(ridge_path, valley_path, origin, radius)

        push!(saddles, saddle)
        push!(ridge_edges_out, SVector{2,Int}(mi, mj))
        push!(valley_edges_out, SVector{2,Int}(va, vb))
        connectors[(:max_max, mi, mj)] = ridge_path
        connectors[(:min_min, va, vb)] = valley_path
    end

    # Fallback: boundary Delaunay edges → old geodesic relaxer
    if !isempty(fallback_del_edges)
        println(
            "  Falling back to geodesic relaxation for $(length(fallback_del_edges)) boundary edge(s).",
        )
        fb_saddles = relax_saddles_geodesic(
            maxima, fallback_del_edges, sampler, origin, radius
        )
        for (fb_k, fb_sad) in enumerate(fb_saddles)
            d_edge = fallback_del_edges[fb_k]
            push!(saddles, fb_sad)
            push!(ridge_edges_out, d_edge)
            push!(valley_edges_out, SVector{2,Int}(0, 0))  # no valley arc
        end
    end

    return saddles, connectors, ridge_edges_out, valley_edges_out
end

"""
    trace_chemical_skeleton(topo::ChemicalTopology, mesh::SphericalScalarMesh, sampler::FieldSampler)

Generates the `:min_max` slope connectors (grey paths) of the 1-skeleton.
Ridge arcs (`:max_max`) and valley arcs (`:min_min`) are already stored in
`topo.connectors` by `optimize_ridge_valley_arcs`; this function only adds the
diagonal slope paths between each saddle's paired maxima and minima.

Spline tuning kwargs (forwarded to `optimize_spline_connector`):
- `threshold`:        Minimum relative improvement over the geodesic to accept a bent arc (default `0.005`).
- `w_proximity`:      δ² penalty weight for control-point displacement from its initial position (default `0.1`).
- `w_mono`:           Unimodal shape penalty weight; not enforced for `:descent` slope paths (default `1.0`).
- `probe_smooth_pct`: Gaussian smoothing fraction for initial control-point placement (default `0.10`).
- `max_iters`:        Maximum refinement iterations per arc (default `5`).
- `conv_tol`:         Convergence tolerance as a fraction of arc length (default `1e-3`).
- `skewness`:         Sample-point density exponent; `> 1` clusters samples near the per-arc
                      saddle (derived from probe extremum each iteration). Default: `2.0`.
"""
function trace_chemical_skeleton(
    topo::ChemicalTopology,
    mesh::SphericalScalarMesh,
    sampler::FieldSampler;
    threshold=0.005,
    w_proximity=0.1,
    w_mono=1.0,
    probe_smooth_pct=0.10,
    max_iters=5,
    conv_tol=1e-3,
    skewness=2.0,
)
    paths = Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}()

    n_sad = length(topo.saddles)
    println("Tracing slope connectors (Min→Max) for $n_sad saddles...")

    for s_idx in 1:n_sad
        r_edge = topo.ridge_edges[s_idx]
        v_edge = topo.valley_edges[s_idx]

        # Skip fallback saddles that have no valley arc connectivity stored
        if v_edge == SVector{2,Int}(0, 0)
            ;
            continue;
        end

        # Slope Paths (Grey): each min → each max that share this saddle
        for mx in (r_edge[1], r_edge[2]), mn in (v_edge[1], v_edge[2])
            key = (:min_max, mn, mx)
            if !haskey(paths, key)
                paths[key] = optimize_spline_connector(
                    topo.maxima[mx],
                    topo.minima[mn],
                    mesh,
                    sampler;
                    type=:descent,
                    threshold=threshold,
                    w_proximity=w_proximity,
                    w_mono=w_mono,
                    probe_smooth_pct=probe_smooth_pct,
                    max_iters=max_iters,
                    conv_tol=conv_tol,
                    skewness=skewness,
                )
            end
        end
    end

    return paths
end

# Tie-breaker helper
function is_less(i, j, values)
    if values[i] != values[j]
        return values[i] < values[j]
    end
    return i < j
end

"""
    predict_topology_voronoi(maxima_coords::Vector{SVector{3, Float64}}, origin, radius)

Generates predicted locations for Minima and Saddles based on the Voronoi dual 
of the given Maxima.

Returns: (predicted_minima, predicted_saddles)
"""
function predict_topology_voronoi(
    maxima::Vector{SVector{3,Float64}}, origin::SVector{3,Float64}, radius::Float64
)
    n = length(maxima)

    # Handle Edge Cases
    if n < 3
        # Logic for N=1, N=2 handled by the "Regime" function upstream.
        return (SVector{3,Float64}[], SVector{2,Int}[], Union{SVector{2,Int},Nothing}[])
    end

    # Store Delaunay triplets (indices of maxima)
    delaunay_triplets = Vector{SVector{3,Int}}()

    # 1. Find Delaunay Triangles
    # A triplet (i, j, k) is Delaunay if its circumcircle on the sphere contains no other points.
    for i in 1:n, j in (i + 1):n, k in (j + 1):n
        p1, p2, p3 = maxima[i], maxima[j], maxima[k]

        # Calculate Normal to the plane defined by the 3 points
        # The circumcenter direction is this normal.
        plane_normal = cross(p2 - p1, p3 - p1)
        norm_val = norm(plane_normal)

        if norm_val < 1e-9
            continue
        end # Collinear points

        dir = plane_normal / norm_val

        # Orient the normal OUTWARD (assuming convex hull property on sphere)
        if dot(dir, p1 - origin) < 0
            dir = -dir
        end

        # Check Empty Circumcircle Property: all other points must be "below" the plane.
        is_delaunay = true
        for m in 1:n
            if m == i || m == j || m == k
                ;
                continue;
            end
            dist = dot(maxima[m] - p1, dir)
            if dist > 1e-7
                is_delaunay = false
                break
            end
        end

        if is_delaunay
            push!(delaunay_triplets, SVector(i, j, k))
        end
    end

    # 2. Compute circumcenters (predicted minima) and build edge→triangle adjacency.
    #    edge_to_tri maps each sorted edge pair to the list of triangle indices
    #    (into delaunay_triplets / pred_minima) that contain it.
    pred_minima = SVector{3,Float64}[]
    edge_to_tri = Dict{SVector{2,Int},Vector{Int}}()

    for (tri_idx, tri) in enumerate(delaunay_triplets)
        p1, p2, p3 = maxima[tri[1]], maxima[tri[2]], maxima[tri[3]]
        normal = normalize(cross(p2 - p1, p3 - p1))
        if dot(normal, p1 - origin) < 0
            ;
            normal = -normal;
        end
        push!(pred_minima, origin + normal * radius)

        for edge_key in (
            SVector{2,Int}(minmax(tri[1], tri[2])),
            SVector{2,Int}(minmax(tri[2], tri[3])),
            SVector{2,Int}(minmax(tri[3], tri[1])),
        )
            v = get!(edge_to_tri, edge_key, Int[])
            push!(v, tri_idx)
        end
    end

    # 3. Build ordered delaunay_edges and dual voronoi_edges.
    #    voronoi_edges[k] = SVector{2,Int}(min_a_idx, min_b_idx) for interior edges,
    #                        nothing for boundary edges (only one adjacent triangle).
    delaunay_edges_vec = SVector{2,Int}[]
    voronoi_edges_vec = Union{SVector{2,Int},Nothing}[]

    for (d_edge, tri_indices) in edge_to_tri
        push!(delaunay_edges_vec, d_edge)
        if length(tri_indices) == 2
            push!(voronoi_edges_vec, SVector{2,Int}(tri_indices[1], tri_indices[2]))
        else
            push!(voronoi_edges_vec, nothing)  # boundary edge — fallback required
        end
    end

    return pred_minima, delaunay_edges_vec, voronoi_edges_vec
end

"""
    clean_voronoi_topology(refined_min, pred_edges, voronoi_edges, origin, radius; tol_factor=0.15)

Removes Voronoi/Delaunay degeneracies **before** arc optimisation so that
downstream steps never see spurious edges or duplicate minima.

Deals with the 4-way (quad) Delaunay co-circularity case where two Voronoi
circumcenters collapse to the same position after relaxation:

1. Merges co-located minima (within `tol_factor * radius`) using greedy
   clustering; projects each cluster centroid back onto the sphere.
2. Drops any `(pred_edge, voronoi_edge)` pair whose Voronoi edge becomes a
   self-loop after remapping (i.e., both endpoints merged into the same
   minimum — this is the spurious quad-diagonal ridge).
3. Deduplicates remaining Delaunay edges that map to the same maxima pair.

Returns `(merged_min, clean_pred_edges, clean_voronoi_edges)` where the two
edge vectors are still parallel and index-aligned with each other, and minima
indices in `clean_voronoi_edges` reference `merged_min`.
"""
function clean_voronoi_topology(
    refined_min::Vector{SVector{3,Float64}},
    pred_edges::Vector{SVector{2,Int}},
    voronoi_edges::Vector{Union{SVector{2,Int},Nothing}},
    origin::SVector{3,Float64},
    radius::Float64;
    tol_factor=0.15,
)
    dist_tol = tol_factor * radius

    # --- Step 1: Merge co-located minima ---
    merged_min = SVector{3,Float64}[]
    old_to_new_min = zeros(Int, length(refined_min))
    used = Set{Int}()

    for i in 1:length(refined_min)
        i in used && continue
        cluster = [refined_min[i]]
        cluster_idx = [i]
        push!(used, i)
        for j in (i + 1):length(refined_min)
            j in used && continue
            if norm(refined_min[i] - refined_min[j]) < dist_tol
                push!(cluster, refined_min[j])
                push!(cluster_idx, j)
                push!(used, j)
            end
        end
        avg = sum(cluster) / length(cluster)
        p_final = origin + normalize(avg - origin) * radius
        push!(merged_min, p_final)
        new_idx = length(merged_min)
        for old_idx in cluster_idx
            old_to_new_min[old_idx] = new_idx
        end
    end

    n_merged = length(refined_min) - length(merged_min)
    println(
        "Voronoi Cleanup: Merged $n_merged duplicate minima " *
        "($(length(refined_min)) → $(length(merged_min))).",
    )

    # --- Step 2: Remap Voronoi edges and drop self-loops / duplicates ---
    clean_pred_edges = SVector{2,Int}[]
    clean_voronoi_edges = Union{SVector{2,Int},Nothing}[]
    seen_delaunay = Set{SVector{2,Int}}()  # deduplicate max–max pairs

    for (i, d_edge) in enumerate(pred_edges)
        # Skip already-seen maxima pairs
        d_sorted = SVector{2,Int}(min(d_edge[1], d_edge[2]), max(d_edge[1], d_edge[2]))
        d_sorted in seen_delaunay && continue

        v_edge = voronoi_edges[i]
        if v_edge === nothing
            # Boundary edge — keep as-is (geodesic fallback will handle it)
            push!(seen_delaunay, d_sorted)
            push!(clean_pred_edges, d_edge)
            push!(clean_voronoi_edges, nothing)
            continue
        end

        # Remap old minimum indices to merged indices
        va = v_edge[1] <= length(old_to_new_min) ? old_to_new_min[v_edge[1]] : v_edge[1]
        vb = v_edge[2] <= length(old_to_new_min) ? old_to_new_min[v_edge[2]] : v_edge[2]

        if va == vb
            # Self-loop: the two Voronoi circumcenters merged → spurious quad-diagonal edge
            println(
                "Voronoi Cleanup: Dropped spurious Delaunay edge $d_edge " *
                "(Voronoi endpoints both merged to minimum $va).",
            )
            continue
        end

        push!(seen_delaunay, d_sorted)
        push!(clean_pred_edges, d_edge)
        v_sorted = SVector{2,Int}(min(va, vb), max(va, vb))
        push!(clean_voronoi_edges, v_sorted)
    end

    return merged_min, clean_pred_edges, clean_voronoi_edges
end

"""
    clean_degenerate_topology(minima, saddles, origin, radius; ridge_edges, valley_edges, tol_factor)

Merges features that have relaxed into the same physical location due to
Delaunay degeneracies (e.g., 4-way symmetric faces).

When `ridge_edges` and `valley_edges` are supplied (aligned 1-to-1 with `saddles`),
they are kept index-aligned with the returned clean saddles, and valley edge min-indices
are remapped to account for merged minima.

Returns `(merged_minima, clean_saddles, clean_ridge_edges, clean_valley_edges)`.
"""
function clean_degenerate_topology(
    minima::Vector{SVector{3,Float64}},
    saddles::Vector{SVector{3,Float64}},
    origin::SVector{3,Float64},
    radius::Float64;
    ridge_edges::Vector{SVector{2,Int}}=SVector{2,Int}[],
    valley_edges::Vector{SVector{2,Int}}=SVector{2,Int}[],
    tol_factor=0.05,
)
    dist_tol = tol_factor * radius
    has_edges = !isempty(ridge_edges)

    # 1. Merge Co-located Minima and build old→new index map
    merged_minima = SVector{3,Float64}[]
    old_to_new_min = zeros(Int, length(minima))  # old idx → new idx (0 = not yet assigned)
    used_mins = Set{Int}()

    for i in 1:length(minima)
        if i in used_mins
            ;
            continue;
        end
        cluster = [minima[i]]
        cluster_idx = [i]
        push!(used_mins, i)
        for j in (i + 1):length(minima)
            if j in used_mins
                ;
                continue;
            end
            if norm(minima[i] - minima[j]) < dist_tol
                push!(cluster, minima[j])
                push!(cluster_idx, j)
                push!(used_mins, j)
            end
        end
        # Project cluster centroid back to sphere
        avg_pos = sum(cluster) / length(cluster)
        p_final = origin + normalize(avg_pos - origin) * radius
        push!(merged_minima, p_final)
        new_idx = length(merged_minima)
        for old_idx in cluster_idx
            old_to_new_min[old_idx] = new_idx
        end
    end

    # 2. Annihilate Spurious Saddles and propagate edge arrays
    clean_saddles = SVector{3,Float64}[]
    clean_ridge_edges = SVector{2,Int}[]
    clean_valley_edges = SVector{2,Int}[]

    for (s_idx, s) in enumerate(saddles)
        is_spurious = any(m -> norm(s - m) < dist_tol, merged_minima)
        if is_spurious
            ;
            continue;
        end

        push!(clean_saddles, s)
        if has_edges
            push!(clean_ridge_edges, ridge_edges[s_idx])
            # Remap valley-edge min indices to post-merge numbering
            v = valley_edges[s_idx]
            if v != SVector{2,Int}(0, 0) &&
                v[1] <= length(old_to_new_min) &&
                v[2] <= length(old_to_new_min)
                push!(
                    clean_valley_edges,
                    SVector{2,Int}(old_to_new_min[v[1]], old_to_new_min[v[2]]),
                )
            else
                push!(clean_valley_edges, v)  # fallback sentinel passes through unchanged
            end
        end
    end

    println(
        "Topology Cleanup: Merged $(length(minima)) minima → $(length(merged_minima)). " *
        "Removed $(length(saddles) - length(clean_saddles)) spurious saddles.",
    )

    return merged_minima, clean_saddles, clean_ridge_edges, clean_valley_edges
end

"""
    compute_discrete_gradient(mesh::SphericalScalarMesh)
"""
function compute_discrete_gradient(mesh::SphericalScalarMesh)
    n_v = length(mesh.values)
    n_e = length(mesh.edges)
    n_t = length(mesh.triangles)

    # 1. Build Adjacency Maps for Lower Star Search
    # We need to know which edges belong to which triangle for the secondary pairing
    t_to_e_map = [Int[] for _ in 1:n_t]

    # Fast Edge Lookup: Dict[Edge, EdgeIndex]
    edge_lookup = Dict{SVector{2,Int},Int}()
    for (i, e) in enumerate(mesh.edges)
        edge_lookup[e] = i
    end

    for (t_idx, tri) in enumerate(mesh.triangles)
        e1 = edge_lookup[SVector{2,Int}(minmax(tri[1], tri[2]))]
        e2 = edge_lookup[SVector{2,Int}(minmax(tri[2], tri[3]))]
        e3 = edge_lookup[SVector{2,Int}(minmax(tri[3], tri[1]))]
        push!(t_to_e_map[t_idx], e1)
        push!(t_to_e_map[t_idx], e2)
        push!(t_to_e_map[t_idx], e3)
    end

    # We also need v -> edges map (already in mesh.connectivity, but we need edge indices)
    v_to_e_list = [Int[] for _ in 1:n_v]
    for (i, e) in enumerate(mesh.edges)
        push!(v_to_e_list[e[1]], i)
        push!(v_to_e_list[e[2]], i)
    end

    # 2. Initialize Pairings
    v_to_e = zeros(Int, n_v)
    e_to_v = zeros(Int, n_e)
    e_to_t = zeros(Int, n_e)
    t_to_e = zeros(Int, n_t)

    processed_e = fill(false, n_e)
    processed_t = fill(false, n_t)

    # 3. Process Lower Stars
    sorted_v = sortperm(mesh.values)

    for v in sorted_v
        # Identify Lower Star (LS)
        # LS Edges: edges incident to v where v is the higher value vertex
        ls_e = Int[]
        for e_idx in v_to_e_list[v]
            e = mesh.edges[e_idx]
            other_v = (e[1] == v) ? e[2] : e[1]
            if is_less(other_v, v, mesh.values) && !processed_e[e_idx]
                push!(ls_e, e_idx)
            end
        end

        # LS Triangles: triangles incident to v where v is the max vertex
        # (We can find these by looking at triangles sharing the LS edges)
        ls_t = Int[]
        # Naive scan of triangles incident to v (could be optimized with v->t map)
        # Given mesh size, we can iterate all triangles or build a v->t map.
        # Let's rely on the edges to find candidates.
        candidate_triangles = Set{Int}()
        for e_idx in ls_e
            # This part requires e->t lookup or we scan t_to_e_map.
            # To keep it simple/pure: we iterate locally.
            # Optimization: Build v->t map during init if this is slow.
        end
        # Actually, let's just use a simple v->t map for speed.
        # (Implementation detail: Add v_to_t_list construction at top if needed)

        # --- Simplified Logic for LS_T ---
        # For now, iterate LS edges, find triangles that contain them.
        # A triangle is in LS(v) if all its vertices <= v (and contains v).
        # We need a robust way to get triangles incident to v.
        # Let's just build v_to_t_list at the start. It's cheap.
    end

    # ... (Re-running logic with v_to_t_list included for correctness) ...

    # Let's restart the function body to include v_to_t_list
    return _compute_gradient_internal(
        mesh, n_v, n_e, n_t, t_to_e_map, v_to_e_list, sorted_v
    )
end

function _compute_gradient_internal(mesh, n_v, n_e, n_t, t_to_e_map, v_to_e_list, sorted_v)
    # Build v -> t map
    v_to_t_list = [Int[] for _ in 1:n_v]
    for (t_idx, tri) in enumerate(mesh.triangles)
        push!(v_to_t_list[tri[1]], t_idx)
        push!(v_to_t_list[tri[2]], t_idx)
        push!(v_to_t_list[tri[3]], t_idx)
    end

    v_to_e = zeros(Int, n_v)
    e_to_v = zeros(Int, n_e)
    e_to_t = zeros(Int, n_e)
    t_to_e = zeros(Int, n_t)

    processed_e = fill(false, n_e)
    processed_t = fill(false, n_t)

    for v in sorted_v
        # 1. Gather LS Edges
        ls_e = Int[]
        for e_idx in v_to_e_list[v]
            if !processed_e[e_idx]
                e = mesh.edges[e_idx]
                other = (e[1] == v) ? e[2] : e[1]
                if is_less(other, v, mesh.values)
                    push!(ls_e, e_idx)
                end
            end
        end

        # 2. Gather LS Triangles
        ls_t = Int[]
        for t_idx in v_to_t_list[v]
            if !processed_t[t_idx]
                tri = mesh.triangles[t_idx]
                # Check if v is the strictly max vertex in this triangle
                if all(u -> (u == v) || is_less(u, v, mesh.values), tri)
                    push!(ls_t, t_idx)
                end
            end
        end

        # 3. Pairing: Vertex -> Edge
        if !isempty(ls_e)
            # Pair v with the steepest descent edge in LS
            # Steepest = edge connected to the neighbor with lowest value
            best_e_idx = -1
            min_val_idx = -1

            for e_idx in ls_e
                e = mesh.edges[e_idx]
                u = (e[1] == v) ? e[2] : e[1]
                if best_e_idx == -1 || is_less(u, min_val_idx, mesh.values)
                    best_e_idx = e_idx
                    min_val_idx = u
                end
            end

            # Pair (v, best_e)
            v_to_e[v] = best_e_idx
            e_to_v[best_e_idx] = v
            processed_e[best_e_idx] = true
        end

        # 4. Pairing: Edge -> Triangle (Process remaining LS elements)
        # We need to pair remaining edges in ls_e with triangles in ls_t
        # This is effectively a matching problem on the LS graph.
        # Simple greedy approach usually works for surface meshes.

        while true
            found_pair = false
            for t_idx in ls_t
                if processed_t[t_idx]
                    ;
                    continue;
                end

                # Find an unpaired edge in LS that belongs to this triangle
                # The triangle has 3 edges. We need one that is in ls_e and !processed
                valid_edge = -1

                # Check the 3 edges of this triangle
                for e_idx in t_to_e_map[t_idx]
                    # We check if this edge is in our current LS set and strictly unpaired
                    # Note: We can check processed_e directly
                    if !processed_e[e_idx] && (e_idx in ls_e)
                        # (Check membership in ls_e is implicit if we trust the topology, 
                        # but explicit check is safer)
                        # Actually, simply checking !processed_e[e_idx] is enough 
                        # because we only iterate ls_t which implies edges are in LS or connecting v.
                        # Wait, the edge MUST be in the Lower Star (connected to v).
                        # Edges opposite to v are NOT in ls_e.

                        # Check if edge contains v (it must to be in ls_e)
                        edge_nodes = mesh.edges[e_idx]
                        if (edge_nodes[1] == v || edge_nodes[2] == v)
                            valid_edge = e_idx
                            break
                        end
                    end
                end

                if valid_edge != -1
                    # Pair (edge, triangle)
                    e_to_t[valid_edge] = t_idx
                    t_to_e[t_idx] = valid_edge
                    processed_e[valid_edge] = true
                    processed_t[t_idx] = true
                    found_pair = true
                end
            end
            if !found_pair
                ;
                break;
            end
        end

        # Mark everything in LS as processed (if not paired, they become critical)
        for e in ls_e
            ;
            processed_e[e] = true;
        end
        for t in ls_t
            ;
            processed_t[t] = true;
        end
    end

    # 4. Extract Critical Cells
    crit_v = findall(x -> v_to_e[x] == 0, 1:n_v)
    crit_e = findall(x -> e_to_v[x] == 0 && e_to_t[x] == 0, 1:n_e)
    crit_t = findall(x -> t_to_e[x] == 0, 1:n_t)

    return DiscreteGradient(v_to_e, e_to_v, e_to_t, t_to_e, crit_v, crit_e, crit_t)
end

"""
    trace_descending_path(grad, mesh, saddle_edge_idx)

Traces the gradient path from a Critical Saddle (Edge) down to Critical Minima.
Returns: (list_of_minima_indices, list_of_paths)
Each 'path' is a vector of [v, e, v, e...] sequence.
"""
function trace_descending_path(
    grad::DiscreteGradient, mesh::SphericalScalarMesh, saddle_idx::Int
)
    # A saddle is an Edge. It has two vertices. 
    # Paths can flow down from either vertex.
    v1, v2 = mesh.edges[saddle_idx]

    paths = Vector{Int}[]
    minima = Int[]

    for start_v in [v1, v2]
        path = Int[] # Store indices in order: v, e, v, e...
        current_v = start_v

        # While current_v is not critical
        while grad.v_to_e[current_v] != 0
            e_next = grad.v_to_e[current_v]
            push!(path, current_v)
            push!(path, e_next)

            # Find the OTHER vertex of this edge
            ev1, ev2 = mesh.edges[e_next]
            next_v = (ev1 == current_v) ? ev2 : ev1
            current_v = next_v
        end

        # We hit a critical vertex (Minimum)
        push!(path, current_v)
        push!(minima, current_v)
        push!(paths, path)
    end

    return minima, paths
end

"""
    reverse_descending_path!(grad, saddle_idx, path)

Cancels a Saddle-Min pair by reversing the gradient path.
"""
function reverse_descending_path!(
    grad::DiscreteGradient, saddle_idx::Int, path::Vector{Int}
)
    # path is: [v0, e1, v1, e2, ..., vk] (where vk is the Min)
    # We want to re-pair so that:
    # v0 pairs with saddle_idx
    # v1 pairs with e1
    # ...
    # vk pairs with ek

    # 1. Pair v0 with the Saddle
    v0 = path[1]
    grad.v_to_e[v0] = saddle_idx
    grad.e_to_v[saddle_idx] = v0

    # 2. Shift the rest
    # Iterate pairs (v_i, e_{i+1})
    # The path vector has odd length (nodes) or even (v,e pairs)? 
    # My trace function pushes v, then e. Ends with v.
    # path: v0, e1, v1, e2, v2 ... vn

    for i in 3:2:length(path)
        v_curr = path[i]   # v1
        e_prev = path[i - 1] # e1

        grad.v_to_e[v_curr] = e_prev
        grad.e_to_v[e_prev] = v_curr
    end

    # Remove from critical sets
    min_idx = path[end]
    filter!(x -> x != min_idx, grad.critical_vertices)
    filter!(x -> x != saddle_idx, grad.critical_edges)
end

"""
    trace_ascending_path(grad, mesh, saddle_idx, edge_to_triangles)

Traces the gradient path UP from a Critical Saddle (Edge) to Critical Maxima (Triangles).
Returns: (list_of_maxima_indices, list_of_paths)
"""
function trace_ascending_path(
    grad::DiscreteGradient,
    mesh::SphericalScalarMesh,
    saddle_idx::Int,
    edge_to_triangles::Vector{Vector{Int}},
)
    # A saddle (Edge) connects to 2 Triangles (on a closed sphere).
    incident_tris = edge_to_triangles[saddle_idx]

    maxima = Int[]
    paths = Vector{Int}[]

    for start_t in incident_tris
        # The path sequence: [e_start, t_next, e_next, t_next...]
        path = Int[]

        current_e = saddle_idx # The critical saddle is the start
        current_t = start_t

        # We need to verify if start_t is actually an *ascending* step.
        # In the gradient, a pair (e, t) means flow goes e -> t.
        # If t is paired with SOME OTHER edge e', the flow continues t -> e' -> t_next...
        # If t is Critical, we stop.

        valid_path = true
        while true
            push!(path, current_e)
            push!(path, current_t)

            # Check if current_t is paired
            paired_e = grad.t_to_e[current_t]

            if paired_e == 0
                # current_t is Critical! We found a Maximum.
                push!(maxima, current_t)
                push!(paths, path)
                break
            end

            # Continue the path: t -> paired_e
            current_e = paired_e

            # From paired_e, go to the OTHER incident triangle
            # (e shares 2 triangles: current_t and next_t)
            tris = edge_to_triangles[current_e]
            if length(tris) != 2
                # Boundary case or error (shouldn't happen on closed sphere)
                valid_path = false
                break
            end

            # Pick the triangle that isn't where we came from
            next_t = (tris[1] == current_t) ? tris[2] : tris[1]
            current_t = next_t
        end
    end

    return maxima, paths
end

"""
    reverse_ascending_path!(grad, saddle_idx, path)

Cancels a Saddle-Max pair by reversing the gradient path.
"""
function reverse_ascending_path!(grad::DiscreteGradient, saddle_idx::Int, path::Vector{Int})
    # Path: [e0, t0, e1, t1 ... tn] (tn is the Max)
    # e0 is the Critical Saddle.

    # 1. Pair the Saddle (e0) with the first Triangle (t0)
    e0 = path[1]
    t0 = path[2]

    grad.e_to_t[e0] = t0
    grad.t_to_e[t0] = e0

    # 2. Shift the rest
    # Original pairs were (e1, t0), (e2, t1)...
    # New pairs will be (e1, t1), (e2, t2)...

    # Iterate pairs (e_i, t_i) starting from index 3
    # path indices: 1=e0, 2=t0, 3=e1, 4=t1 ...
    for i in 3:2:length(path)
        e_curr = path[i]   # e1
        t_curr = path[i + 1] # t1

        grad.e_to_t[e_curr] = t_curr
        grad.t_to_e[t_curr] = e_curr
    end

    # Remove from critical sets
    max_idx = path[end]
    filter!(x -> x != max_idx, grad.critical_triangles)
    filter!(x -> x != saddle_idx, grad.critical_edges)
end

"""
    simplify_persistence!(grad, mesh, threshold)

Iteratively cancels critical pairs with persistence < threshold.
"""
function simplify_persistence!(
    grad::DiscreteGradient, mesh::SphericalScalarMesh, threshold::Float64
)
    # println("Starting simplification (Threshold: $threshold)...")

    # --- Precompute Edge -> Triangle Adjacency ---
    # We need this for ascending paths.
    n_e = length(mesh.edges)
    edge_to_triangles = [Int[] for _ in 1:n_e]

    # Helper to look up edge indices (reusing the logic from compute_gradient)
    edge_lookup = Dict{SVector{2,Int},Int}()
    for (i, e) in enumerate(mesh.edges)
        edge_lookup[e] = i
    end

    for (t_idx, tri) in enumerate(mesh.triangles)
        for pair in [(tri[1], tri[2]), (tri[2], tri[3]), (tri[3], tri[1])]
            s_pair = SVector{2,Int}(minmax(pair[1], pair[2]))
            e_idx = edge_lookup[s_pair]
            push!(edge_to_triangles[e_idx], t_idx)
        end
    end
    # ---------------------------------------------

    total_cancelled = 0

    while true
        found_any = false

        # --- Pass 1: Min-Saddle Cancellation ---
        best_min_pair = nothing
        min_persist_val = Inf # Use a separate var to track value

        for s_idx in grad.critical_edges
            # Persistence = f(saddle) - f(min)
            # Saddle Value: max(v1, v2)
            v1, v2 = mesh.edges[s_idx]
            val_s = max(mesh.values[v1], mesh.values[v2])

            mins, paths = trace_descending_path(grad, mesh, s_idx)

            for (i, m_idx) in enumerate(mins)
                val_m = mesh.values[m_idx]
                p = val_s - val_m

                # Check uniqueness (only cancel if unique path exists)
                if p < threshold && p < min_persist_val
                    if count(x->x==m_idx, mins) == 1
                        min_persist_val = p
                        best_min_pair = (s_idx, paths[i], :min)
                    end
                end
            end
        end

        # --- Pass 2: Max-Saddle Cancellation ---
        best_max_pair = nothing
        max_persist_val = Inf

        for s_idx in grad.critical_edges
            # Persistence = f(max) - f(saddle)
            v1, v2 = mesh.edges[s_idx]
            val_s = max(mesh.values[v1], mesh.values[v2])

            maxs, paths = trace_ascending_path(grad, mesh, s_idx, edge_to_triangles)

            for (i, mx_idx) in enumerate(maxs)
                # Value of Max Triangle = max value of its vertices?
                # Standard definition: Max value of the simplex.
                # For a triangle, it's the max value of its 3 vertices.
                tv1, tv2, tv3 = mesh.triangles[mx_idx]
                val_mx = max(mesh.values[tv1], mesh.values[tv2], mesh.values[tv3])

                p = val_mx - val_s

                if p < threshold && p < max_persist_val
                    if count(x->x==mx_idx, maxs) == 1
                        max_persist_val = p
                        best_max_pair = (s_idx, paths[i], :max)
                    end
                end
            end
        end

        # --- Execute the Single Best Move ---
        # We pick the lowest persistence pair from EITHER Min or Max side
        # to ensure we simplify the "noise" uniformly.

        target = nothing

        if best_min_pair !== nothing && best_max_pair !== nothing
            if min_persist_val < max_persist_val
                target = best_min_pair
            else
                target = best_max_pair
            end
        elseif best_min_pair !== nothing
            target = best_min_pair
        elseif best_max_pair !== nothing
            target = best_max_pair
        end

        # println("Cancelling pair with persistence $max_persist_val")

        if target !== nothing
            s_idx, path, type = target
            if type == :min
                reverse_descending_path!(grad, s_idx, path)
            else
                reverse_ascending_path!(grad, s_idx, path)
            end
            found_any = true
            total_cancelled += 1
        else
            break # No more valid moves
        end
    end

    # println("Simplification Complete. Removed $total_cancelled pairs.")
end

"""
    suggest_persistence_threshold(mesh, grad; plot=false)

Analyzes the persistence of all critical pairs (Min-Saddle and Max-Saddle) 
to find the "Noise/Signal" gap. Returns a suggested threshold value.
"""
function suggest_persistence_threshold(
    mesh::SphericalScalarMesh, grad::DiscreteGradient; plot=false
)

    # 1. Precompute Edge -> Triangle Adjacency (Required for Ascending Trace)
    # ---------------------------------------------------------------------
    n_e = length(mesh.edges)
    edge_to_triangles = [Int[] for _ in 1:n_e]

    # Fast Edge Lookup
    edge_lookup = Dict{SVector{2,Int},Int}()
    for (i, e) in enumerate(mesh.edges)
        edge_lookup[e] = i
    end

    for (t_idx, tri) in enumerate(mesh.triangles)
        for pair in [(tri[1], tri[2]), (tri[2], tri[3]), (tri[3], tri[1])]
            s_pair = SVector{2,Int}(minmax(pair[1], pair[2]))
            if haskey(edge_lookup, s_pair)
                e_idx = edge_lookup[s_pair]
                push!(edge_to_triangles[e_idx], t_idx)
            end
        end
    end
    # ---------------------------------------------------------------------

    persistences = Float64[]

    # 2. Analyze Min-Saddle Pairs (Descending)
    for s_idx in grad.critical_edges
        v1, v2 = mesh.edges[s_idx]
        # Saddle value approx as max of its vertices
        val_s = max(mesh.values[v1], mesh.values[v2])

        mins, _ = trace_descending_path(grad, mesh, s_idx)

        for m_idx in mins
            val_m = mesh.values[m_idx]
            p = val_s - val_m
            if p > 1e-9
                ;
                push!(persistences, p);
            end
        end
    end

    # 3. Analyze Max-Saddle Pairs (Ascending)
    for s_idx in grad.critical_edges
        v1, v2 = mesh.edges[s_idx]
        val_s = max(mesh.values[v1], mesh.values[v2])

        maxs, _ = trace_ascending_path(grad, mesh, s_idx, edge_to_triangles)

        for mx_idx in maxs
            # Max value is max of triangle vertices
            t_verts = mesh.triangles[mx_idx]
            val_mx = max(
                mesh.values[t_verts[1]], mesh.values[t_verts[2]], mesh.values[t_verts[3]]
            )

            p = val_mx - val_s
            if p > 1e-9
                ;
                push!(persistences, p);
            end
        end
    end

    # 4. Analyze the Gap (The Elbow Method)
    sort!(persistences)

    if isempty(persistences)
        println("No persistence pairs found.")
        return 0.0
    end

    # Use Log-Space differences to find orders-of-magnitude gaps
    # Filter out near-zero floating point artifacts
    p_clean = filter(x -> x > 1e-9, persistences)

    if isempty(p_clean)
        return 1e-6 # Fallback small threshold
    end

    log_p = log10.(p_clean)
    gaps = diff(log_p)

    # Find the single largest jump in persistence scale
    max_gap_idx = argmax(gaps)

    noise_ceil = p_clean[max_gap_idx]
    signal_floor = p_clean[max_gap_idx + 1]

    # Suggest the geometric mean between noise and signal
    suggested_threshold = sqrt(noise_ceil * signal_floor)

    if plot
        println("\n--- Persistence Analysis ---")
        println("Total Pairs Checked: $(length(persistences))")
        println("Clean Pairs (>1e-7): $(length(p_clean))")
        println("Noise Ceiling:       $(round(noise_ceil, sigdigits=4))")
        println("Signal Floor:        $(round(signal_floor, sigdigits=4))")
        println("Largest Gap (log10): $(round(gaps[max_gap_idx], sigdigits=3))")
        println("Suggested Threshold: $(round(suggested_threshold, sigdigits=4))")
        println("----------------------------\n")
    end

    return suggested_threshold
end

"""
    identify_primary_anchors(mesh, grad; min_drop_ratio=2.0)

Identifies the significant "Anchor" maxima by analyzing the prominence drop.
Returns: Vector of *Refined* SVector{3, Float64} coordinates.
"""
function identify_primary_anchors(
    mesh::SphericalScalarMesh,
    grad::DiscreteGradient;
    min_drop_ratio=2.0,
    merge_tol_factor=0.15,
)

    # --- Helper: Quadratic Refinement (Inline) ---
    function refine_max_from_triangle(t_idx::Int)
        t_verts = mesh.triangles[t_idx]
        seed_node = t_verts[argmax([mesh.values[v] for v in t_verts])]

        neighbors = mesh.connectivity[seed_node]
        if isempty(neighbors)
            ;
            return mesh.positions[seed_node];
        end

        origin = mesh.positions[seed_node]
        normal = normalize(origin - mesh.origin)
        arb = abs(normal[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
        u = normalize(cross(normal, arb))
        v = cross(normal, u)

        points_uv = SVector{2,Float64}[]
        push!(points_uv, SVector(0.0, 0.0))
        values = Float64[mesh.values[seed_node]]

        for n_idx in neighbors
            diff = mesh.positions[n_idx] - origin
            push!(points_uv, SVector(dot(diff, u), dot(diff, v)))
            push!(values, mesh.values[n_idx])
        end

        if length(values) < 6
            ;
            return origin;
        end

        A = zeros(length(values), 6)
        for (i, p) in enumerate(points_uv)
            A[i, 1] = p[1]^2;
            A[i, 2] = p[2]^2;
            A[i, 3] = p[1]*p[2]
            A[i, 4] = p[1];
            A[i, 5] = p[2];
            A[i, 6] = 1.0
        end

        coeffs = A \ values
        M = [2*coeffs[1] coeffs[3]; coeffs[3] 2*coeffs[2]]
        RHS = [-coeffs[4], -coeffs[5]]

        if det(M) ≈ 0
            ;
            return origin;
        end

        uv_opt = M \ RHS

        d_max = maximum(norm.(points_uv))
        if norm(uv_opt) > d_max
            uv_opt = normalize(uv_opt) * d_max
        end

        pos_opt = origin + uv_opt[1]*u + uv_opt[2]*v
        return mesh.origin + normalize(pos_opt - mesh.origin) * mesh.radius
    end
    # ---------------------------------------------

    # 1. Collect All Candidates
    raw_candidates = []

    for t_idx in grad.critical_triangles
        t = mesh.triangles[t_idx]
        val = max(mesh.values[t[1]], mesh.values[t[2]], mesh.values[t[3]])

        # Approximate coordinate for distance checking
        p = (mesh.positions[t[1]] + mesh.positions[t[2]] + mesh.positions[t[3]]) / 3.0

        push!(raw_candidates, (index=t_idx, value=val, coord=p))
    end

    # 2. Sort Descending by Value
    sort!(raw_candidates; by=x -> x.value, rev=true)

    if isempty(raw_candidates)
        ;
        return SVector{3,Float64}[];
    end

    # 3. Spatial Non-Maximum Suppression (NMS)
    candidates = []
    dist_tol = merge_tol_factor * mesh.radius

    for cand in raw_candidates
        is_redundant = false
        # Check against already accepted (higher value) candidates
        for accepted in candidates
            if norm(cand.coord - accepted.coord) < dist_tol
                is_redundant = true
                break
            end
        end

        if !is_redundant
            push!(candidates, cand)
        end
    end

    println("\n--- Anchor Filtering ---")
    println(
        "Spatial NMS: Clustered $(length(raw_candidates)) raw maxima down to $(length(candidates)) unique peaks.",
    )

    # 4. Analyze Prominence (The "Knee" Finding)
    cutoff_idx = length(candidates)
    check_limit = min(length(candidates) - 1, 20)
    best_gap = 0.0

    for i in 1:check_limit
        val_curr = candidates[i].value
        val_next = candidates[i + 1].value

        if val_next < 1e-9
            ;
            val_next = 1e-9;
        end

        ratio = val_curr / val_next
        println(
            "Anchor $i ($val_curr) -> Anchor $(i+1) ($val_next): Drop Ratio = $(round(ratio, digits=2))",
        )

        if ratio > best_gap && ratio > min_drop_ratio
            best_gap = ratio
            cutoff_idx = i
        end
    end

    println("Selected Top $cutoff_idx Anchors (Gap: $(round(best_gap, digits=2))x)")
    println("------------------------")

    # 5. Refine and Return
    refined_anchors = SVector{3,Float64}[]
    for i in 1:cutoff_idx
        # Now we only pay the cost of the quadratic fit on the true cluster centers
        p = refine_max_from_triangle(candidates[i].index)
        push!(refined_anchors, p)
    end

    return refined_anchors
end

"""
    anisotropic_smooth(mesh::SphericalScalarMesh; 
                       iterations::Int=10, 
                       lambda::Float64=0.1, 
                       k::Float64=0.01)

Applies Perona-Malik anisotropic diffusion to the scalar field.

# Arguments
- `lambda` (0.0-0.25): Integration constant (speed of diffusion). Keep distinct < 0.25 for stability.
- `k`: Gradient threshold. 
    - Differences > k are preserved (considered "edges").
    - Differences < k are smoothed (considered "noise").
"""
function anisotropic_smooth(
    mesh::SphericalScalarMesh; iterations::Int=10, lambda::Float64=0.1, k::Float64=0.01
)

    # Copy values to avoid mutating the original mesh immediately (buffer for updates)
    current_values = copy(mesh.values)
    next_values = copy(mesh.values)
    n_v = length(current_values)

    # Pre-compute inverse distances for speed (optional, but good for accuracy)
    # If the mesh is very uniform, you can skip this and treat dist=1.
    # Storing this as a Vector of Vectors to match connectivity.
    inv_dists = Vector{Vector{Float64}}(undef, n_v)
    for i in 1:n_v
        neighbors = mesh.connectivity[i]
        dists = Float64[]
        p_i = mesh.positions[i]
        for neighbor in neighbors
            d = great_circle_distance(p_i, mesh.positions[neighbor], mesh.origin)
            push!(dists, d > 0 ? 1.0/d : 0.0)
        end
        inv_dists[i] = dists
    end

    # Diffusion Kernel
    # g(diff) = exp(-(diff/k)^2)
    function conduction_coeff(diff, k)
        return exp(-(diff / k)^2)
    end

    for iter in 1:iterations
        # Threads.@threads for i in 1:n_v # Uncomment for parallel speedup on large meshes
        for i in 1:n_v
            flux_sum = 0.0

            neighbors = mesh.connectivity[i]
            dists = inv_dists[i]
            val_i = current_values[i]

            for (j_idx, neighbor_id) in enumerate(neighbors)
                val_j = current_values[neighbor_id]

                # Gradient approximation
                delta = val_j - val_i
                dist_weight = dists[j_idx] # 1/dx

                # Physical gradient magnitude ~ delta / dx
                grad_mag = abs(delta) * dist_weight

                # Conduction coefficient (0 to 1)
                c = conduction_coeff(grad_mag, k)

                # Diffusion update
                flux_sum += c * delta # standard Laplacian would just be 'delta'
            end

            next_values[i] = val_i + lambda * flux_sum
        end

        # Swap buffers
        current_values .= next_values
    end

    # Return a NEW mesh with smoothed values (keep geometry/topo same)
    return SphericalScalarMesh(
        mesh.positions,
        mesh.radius,
        mesh.origin,
        mesh.connectivity,
        mesh.triangles,
        mesh.edges,
        current_values, # The smoothed signal
    )
end

"""
    smooth_mesh_hybrid(mesh::SphericalScalarMesh; 
                       median_iters::Int=1, 
                       laplacian_iters::Int=5, 
                       lambda::Float64=0.5)

Applies a Median filter (to kill spikes) followed by Laplacian smoothing (to regularize gradients).

# Arguments
- `median_iters`: Number of median passes. Usually 1 is enough to kill spikes.
- `laplacian_iters`: Number of smoothing steps.
- `lambda`: Smoothing strength (0.0 - 1.0).
"""
function smooth_mesh_hybrid(
    mesh::SphericalScalarMesh;
    median_iters::Int=1,
    laplacian_iters::Int=5,
    lambda::Float64=0.5,
)
    current_values = copy(mesh.values)
    next_values = copy(mesh.values)
    n_v = length(current_values)

    # --- Phase 1: Median Filter (The "Spike Killer") ---
    # Great for "large, spurious fluctuations" that distort gradients.
    for iter in 1:median_iters
        for i in 1:n_v
            neighbors = mesh.connectivity[i]

            # Collect values: self + neighbors
            # (Allocation-heavy? For 100k nodes, reusing a buffer is better, 
            # but for simplicity/clarity we allocate small vectors here)
            window = Float64[]
            push!(window, current_values[i])
            for n_idx in neighbors
                push!(window, current_values[n_idx])
            end

            next_values[i] = median(window)
        end
        current_values .= next_values
    end

    # --- Phase 2: Geometric Laplacian (The "Contour Smoother") ---
    # Essential to make the field differentiable again after Median steps.
    # Uses cotangent weights or simple inverse-distance weights.
    # Simple inverse-distance is usually sufficient for "uniform-ish" spherical meshes.

    # Precompute weights (optional optimization)
    # Using simple normalized umbrella operator here

    for iter in 1:laplacian_iters
        for i in 1:n_v
            neighbors = mesh.connectivity[i]
            val_i = current_values[i]

            sum_vals = 0.0
            sum_weights = 0.0

            p_i = mesh.positions[i]

            for n_idx in neighbors
                # Inverse Euclidean distance weight
                dist = norm(mesh.positions[n_idx] - p_i)
                w = dist > 1e-9 ? 1.0 / dist : 0.0

                sum_vals += current_values[n_idx] * w
                sum_weights += w
            end

            if sum_weights > 0
                avg_neighbor = sum_vals / sum_weights
                # Update: move towards average
                next_values[i] = val_i + lambda * (avg_neighbor - val_i)
            end
        end
        current_values .= next_values
    end

    return SphericalScalarMesh(
        mesh.positions,
        mesh.radius,
        mesh.origin,
        mesh.connectivity,
        mesh.triangles,
        mesh.edges,
        current_values,
    )
end

function plot_chemical_graph(
    mesh::SphericalScalarMesh, topo::ChemicalTopology; title="Chemical Graph"
)
    traces = GenericTrace[]

    # 1. Surface (Ghost)
    push!(
        traces,
        mesh3d(;
            x=[p[1] for p in mesh.positions],
            y=[p[2] for p in mesh.positions],
            z=[p[3] for p in mesh.positions],
            i=[t[1]-1 for t in mesh.triangles],
            j=[t[2]-1 for t in mesh.triangles],
            k=[t[3]-1 for t in mesh.triangles],
            intensity=mesh.values,
            colorscale="Viridis",
            opacity=1.0,
            showscale=false,
            name="Density",
        ),
    )

    # 2. Critical Points
    function add_cp(coords, color, name, sym, sz)
        if !isempty(coords)
            push!(
                traces,
                scatter3d(;
                    x=[p[1] for p in coords],
                    y=[p[2] for p in coords],
                    z=[p[3] for p in coords],
                    mode="markers",
                    marker=attr(; size=sz, color=color, symbol=sym),
                    name=name,
                ),
            )
        end
    end

    add_cp(topo.maxima, "red", "Maxima (Bonds)", "circle", 6)
    add_cp(topo.minima, "blue", "Minima (Cage)", "circle", 5)
    add_cp(topo.saddles, "green", "Saddles (Ring)", "diamond", 4)

    # 3. Skeleton Paths
    # Helper to batch lines for performance
    function add_paths(filter_type, color, name, width)
        lx, ly, lz = Float64[], Float64[], Float64[]
        for (key, path) in topo.connectors
            if key[1] == filter_type
                for p in path
                    ;
                    push!(lx, p[1]);
                    push!(ly, p[2]);
                    push!(lz, p[3]);
                end
                push!(lx, NaN);
                push!(ly, NaN);
                push!(lz, NaN); # Break line
            end
        end
        if !isempty(lx)
            push!(
                traces,
                scatter3d(;
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    line=attr(; color=color, width=width),
                    name=name,
                ),
            )
        end
    end

    add_paths(:max_max, "red", "Ridges (Max→Max)", 5)
    add_paths(:min_min, "royalblue", "Valleys (Min→Min)", 5)
    add_paths(:min_max, "grey", "Slope", 2) # Thinner, subtle

    layout = Layout(; title=title, scene=attr(; aspectmode="data"), showlegend=true)
    plot(traces, layout)
end

"""
    analyze_condensed_density(mesh::SphericalScalarMesh;
                              smooth_iters, persistence_factor,
                              spline_threshold, spline_w_proximity, spline_w_mono,
                              spline_probe_smooth_pct, spline_max_iters, spline_conv_tol,
                              spline_skewness)

Full Topological Analysis Pipeline:
1. Smoothing (Hybrid Median/Laplacian)
2. Anchor Identification (Persistence & Prominence)
3. Geometric Prediction (Voronoi/Antipodal)
4. Arc Optimization & Saddle Intersection
5. Skeleton Slope Tracing

# Keyword Arguments
- `smooth_iters`:            Number of Laplacian smoothing passes after the median filter (default `10`).
- `persistence_factor`:      Minimum prominence-to-drop ratio for anchor (maxima) identification
                             (default `2.0`).
- `spline_threshold`:        Minimum relative improvement over the geodesic to accept a bent arc.
                             Lower → more curvature; higher → geodesic-like arcs. Default: `0.005`.
- `spline_w_proximity`:      δ² penalty weight on Bézier control-point displacement from its
                             initial on-arc position. Higher → smaller bends; lower → more freedom
                             to follow ridges/valleys. Default: `0.1`.
- `spline_w_mono`:           Unimodal shape penalty weight for ridge (∪) and valley (∩) arcs.
                             Higher → stronger enforcement of single-interior-extremum shape.
                             Default: `1.0`.
- `spline_probe_smooth_pct`: Gaussian kernel half-width (fraction of n_probe) used to smooth field
                             values along the probed path before selecting the initial control
                             point. Default: `0.10`.
- `spline_max_iters`:        Maximum refinement iterations per arc. Default: `5`.
- `spline_conv_tol`:         Convergence tolerance as a fraction of arc length; iteration stops
                             when the control point moves less than this. Default: `1e-2`.
- `spline_skewness`:         Sample-point density exponent for the path-integral objective.
                             `1.0` — uniform; `> 1.0` — denser near each arc's per-arc saddle
                             (derived from the probe extremum), reducing endpoint bias. Default: `2.0`.
"""
function analyze_condensed_density(
    mesh::SphericalScalarMesh;
    smooth_iters=15,
    persistence_factor=2.0,
    spline_threshold=0.001,
    spline_w_proximity=0.005,
    spline_w_mono=1.0,
    spline_probe_smooth_pct=0.10,
    spline_max_iters=5,
    spline_conv_tol=1e-2,
    spline_skewness=2.0,
)
    println("--- 1. Smoothing Mesh ---")
    range = maximum(mesh.values) - minimum(mesh.values)
    smooth_mesh = anisotropic_smooth(mesh; iterations=20, lambda=0.05, k=0.02 * range)
    smooth_mesh = smooth_mesh_hybrid(
        smooth_mesh; median_iters=1, laplacian_iters=smooth_iters
    )
    sampler = build_sampler(smooth_mesh)

    println("--- 2. Discrete Topology Analysis ---")
    grad = compute_discrete_gradient(smooth_mesh)
    p_thresh = suggest_persistence_threshold(smooth_mesh, grad)
    simplify_persistence!(grad, smooth_mesh, p_thresh)

    # 3. Identify Anchors (Refined Maxima)
    anchors = identify_primary_anchors(smooth_mesh, grad; min_drop_ratio=persistence_factor)
    n_anchors = length(anchors)
    println("Identified $n_anchors primary anchors.")

    # 4. Regime Detection & Prediction
    pred_min = SVector{3,Float64}[]
    pred_edges = SVector{2,Int}[]
    voronoi_edges = Union{SVector{2,Int},Nothing}[]

    if n_anchors >= 3
        println("Regime: Molecular (Voronoi Prediction)")
        pred_min, pred_edges, voronoi_edges = predict_topology_voronoi(
            anchors, smooth_mesh.origin, smooth_mesh.radius
        )

    elseif n_anchors == 1
        println("Regime: Terminal (Antipodal Prediction)")
        v_max = anchors[1] - smooth_mesh.origin
        push!(pred_min, smooth_mesh.origin - v_max)  # Antipode

    elseif n_anchors == 2
        println("Regime: Linear/Diatomic (Manifold Analysis)")
        # Placeholder — no Voronoi prediction for 2-anchor regime
    end

    println("--- 3. Elastic Relaxation & Arc Optimization ---")

    # Relax predicted minima (provides good valley-arc endpoints)
    refined_min = relax_minima_constrained(
        pred_min, sampler, smooth_mesh.origin, smooth_mesh.radius; stiffness=1.0
    )

    # Remove Voronoi/Delaunay degeneracies BEFORE arc optimisation so that:
    #  - Spurious quad-diagonal Delaunay edges (whose dual Voronoi edge becomes a
    #    self-loop after merging coincident minima) are dropped entirely.
    #  - Arc optimisation only ever sees the true, deduplicated topology.
    #  - Connector keys and valley-edge indices are already on the merged basis,
    #    so no post-hoc remapping is required.
    merged_min, clean_pred_edges, clean_voronoi_edges = clean_voronoi_topology(
        refined_min,
        pred_edges,
        voronoi_edges,
        smooth_mesh.origin,
        smooth_mesh.radius;
        tol_factor=0.15,
    )

    # Optimize full ridge arcs (max→max) and valley arcs (min→min); derive saddles
    # as their intersections. Boundary edges fall back to geodesic relaxation.
    arc_saddles, arc_connectors, ridge_edges_raw, valley_edges_raw = optimize_ridge_valley_arcs(
        anchors,
        merged_min,
        clean_pred_edges,
        clean_voronoi_edges,
        sampler,
        smooth_mesh.origin,
        smooth_mesh.radius;
        threshold=spline_threshold,
        w_proximity=spline_w_proximity,
        w_mono=spline_w_mono,
        probe_smooth_pct=spline_probe_smooth_pct,
        max_iters=spline_max_iters,
        conv_tol=spline_conv_tol,
        skewness=spline_skewness,
    )

    # Clean degenerate saddles and remap any remaining edge indices.
    # Minima are already merged, so this is mainly a saddle-proximity pass.
    clean_min, clean_sad, clean_ridge_edges, clean_valley_edges = clean_degenerate_topology(
        merged_min,
        arc_saddles,
        smooth_mesh.origin,
        smooth_mesh.radius;
        ridge_edges=ridge_edges_raw,
        valley_edges=valley_edges_raw,
        tol_factor=0.15,
    )

    # Construct Topology Object (pre-populate ridge/valley arcs from optimization)
    topo = ChemicalTopology(
        anchors, clean_min, clean_sad, clean_ridge_edges, clean_valley_edges, arc_connectors
    )

    println("--- 4. Tracing Slope Connectors ---")
    # Append :min_max slope paths; ridge/valley arcs are already in topo.connectors
    slope_paths = trace_chemical_skeleton(
        topo,
        smooth_mesh,
        sampler;
        threshold=spline_threshold,
        w_proximity=spline_w_proximity,
        w_mono=spline_w_mono,
        probe_smooth_pct=spline_probe_smooth_pct,
        max_iters=spline_max_iters,
        conv_tol=spline_conv_tol,
        skewness=spline_skewness,
    )
    merge!(topo.connectors, slope_paths)

    return topo, smooth_mesh
end

function process_sphere(input_path::String)
    spherical_mesh = deserialize(open(input_path, "r"))

    # Print basic stats about the mesh for sanity checking (min/max/mean/stddev of values, number of nodes, etc.)
    println("\n--- Loaded Spherical Mesh ---")
    println("Number of Nodes: $(length(spherical_mesh.positions))")
    println(
        "Value Range    : $(minimum(spherical_mesh.values)) to $(maximum(spherical_mesh.values))",
    )
    println("Mean Value     : $(mean(spherical_mesh.values))")
    println("Std Dev Value  : $(std(spherical_mesh.values))")
    println("Origin         : $(spherical_mesh.origin)")
    println("Radius         : $(spherical_mesh.radius)")
    println("-----------------------------\n")

    # 2. Run Analysis
    topo, smooth_mesh = analyze_condensed_density(spherical_mesh)

    # # Check the length of the first 'Slope' (Min-Max) connector
    # println("Number of connectosrs: $(length(topo.connectors))")
    # for (k, path) in topo.connectors
    #     println("Slope Path Points: $(length(path))")
    # end

    # input_file_basename = splitext(basename(input_path))[1]
    # # Optional consistency check against sibling TSV (same basename)
    # tsv_path = joinpath(dirname(input_path), "$(input_file_basename).tsv")
    # if isfile(tsv_path)
    #     compare_spherical_mesh_to_tsv(spherical_mesh, tsv_path)
    # else
    #     println("[INFO] TSV comparison skipped (file not found): $tsv_path")
    # end

    # 3. Visualize
    display(plot_chemical_graph(spherical_mesh, topo))
    display(plot_chemical_graph(smooth_mesh, topo))

    return spherical_mesh
end

function main(
    input_path::String="/Users/haiiro/scratch/BAND_cleavage/tape41s/Mo4_000_restart_fine_atom_2_V_spherical_mesh.jls",
)
    if !isfile(input_path)
        error("File does not exist: $input_path")
    end

    # process_sphere(input_path)

    return process_sphere(input_path)
end
