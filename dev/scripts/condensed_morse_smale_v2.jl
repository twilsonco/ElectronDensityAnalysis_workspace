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

    # The 1-Skeleton (Optimized Splines)
    # Stored as vectors of control points or high-res sample points
    # Key = (Type, Index1, Index2)
    connectors::Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}
end

# A helper for the Voronoi Prediction results
struct VoronoiDual
    delaunay_triplets::Vector{SVector{3,Int}} # Indices into maxima
    voronoi_vertices::Vector{SVector{3,Float64}} # Predicted Minima
    voronoi_edge_midpoints::Vector{SVector{3,Float64}} # Predicted Saddles
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

"""
    refine_critical_points(mesh, discrete_grad)

Refines the location of critical points by fitting a local quadratic surface 
to the neighborhood of the mesh node closest to the critical cell's centroid.
"""
function refine_critical_points(mesh::SphericalScalarMesh, grad::DiscreteGradient)

    # --- Internal Helper: Quadratic Refinement on a Node Neighborhood ---
    function refine_from_node(node_idx::Int)
        # 1. Gather Neighborhood (One-ring)
        neighbors = mesh.connectivity[node_idx]
        if isempty(neighbors)
            ;
            return mesh.positions[node_idx];
        end

        # 2. Define Local Tangent Frame
        origin = mesh.positions[node_idx]
        normal = normalize(origin - mesh.origin)

        arb = abs(normal[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
        u = normalize(cross(normal, arb))
        v = cross(normal, u)

        # 3. Project neighbors to (u, v)
        # Fit f(u,v) = c1*u^2 + c2*v^2 + c3*u*v + c4*u + c5*v + c6
        points_uv = SVector{2,Float64}[]
        push!(points_uv, SVector(0.0, 0.0)) # Center
        values = Float64[mesh.values[node_idx]]

        for n_idx in neighbors
            diff = mesh.positions[n_idx] - origin
            push!(points_uv, SVector(dot(diff, u), dot(diff, v)))
            push!(values, mesh.values[n_idx])
        end

        if length(values) < 6
            return origin
        end

        # Build Least Squares Matrix
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

        # 4. Find Analytical Extremum
        M = [2*coeffs[1] coeffs[3]; coeffs[3] 2*coeffs[2]]
        RHS = [-coeffs[4], -coeffs[5]]

        if det(M) ≈ 0
            ;
            return origin;
        end

        uv_opt = M \ RHS

        # Clamp to avoid wild extrapolation (e.g. limit to 2x avg edge length)
        # Simple heuristic: if magnitude > distance to nearest neighbor, clamp.
        d_max = maximum(norm.(points_uv))
        if norm(uv_opt) > d_max
            uv_opt = normalize(uv_opt) * d_max
        end

        # 5. Project back to 3D Sphere
        pos_opt = origin + uv_opt[1]*u + uv_opt[2]*v
        return mesh.origin + normalize(pos_opt - mesh.origin) * mesh.radius
    end

    # --- Element-to-Node Conversion Logic ---
    function get_seed_node(index::Int, type::Symbol)
        if type == :vertex
            return index # It's already a node index

        elseif type == :edge
            # For an edge, pick the vertex with the more extreme value?
            # Or just the first one. Let's pick the one with the higher value 
            # if we are looking for a max-ish thing, lower for min-ish.
            # Saddles are ambiguous. Let's just pick v1.
            # Better: pick the vertex closer to the edge midpoint? (Trivial, they are equidistant)
            v1, v2 = mesh.edges[index]
            return v1

        elseif type == :triangle
            # For a triangle, pick the vertex with the max value (since it's a Maximum)
            v1, v2, v3 = mesh.triangles[index]
            vals = (mesh.values[v1], mesh.values[v2], mesh.values[v3])
            # argmax
            if vals[1] >= vals[2] && vals[1] >= vals[3]
                return v1
            elseif vals[2] >= vals[1] && vals[2] >= vals[3]
                return v2
            else
                return v3
            end
        end
    end

    # --- Execute Refinement ---
    # 1. Maxima (Critical Triangles)
    refined_max = SVector{3,Float64}[]
    for t_idx in grad.critical_triangles
        seed = get_seed_node(t_idx, :triangle)
        push!(refined_max, refine_from_node(seed))
    end

    # 2. Minima (Critical Vertices)
    refined_min = SVector{3,Float64}[]
    for v_idx in grad.critical_vertices
        seed = get_seed_node(v_idx, :vertex)
        push!(refined_min, refine_from_node(seed))
    end

    # 3. Saddles (Critical Edges)
    refined_sad = SVector{3,Float64}[]
    for e_idx in grad.critical_edges
        seed = get_seed_node(e_idx, :edge)
        push!(refined_sad, refine_from_node(seed))
    end

    # Return empty dict for connectors (to be filled later)
    return ChemicalTopology(
        refined_max,
        refined_min,
        refined_sad,
        Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}(),
    )
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

function path_integral_objective(
    controls_flat,
    p1,
    p2,
    sampler,
    origin,
    radius,
    type;
    samples=20,
    w_reg=0.1,
    w_radial=10.0,
) # Added w_radial
    n_ctrl = length(controls_flat) ÷ 3
    controls = SVector{3,Float64}[]

    # 1. Reconstruct Control Points & Calculate Radial Penalty
    radial_penalty = 0.0

    for i in 1:n_ctrl
        slice = controls_flat[(3 * i - 2):(3 * i)]
        c = SVector{3,Float64}(slice[1], slice[2], slice[3])
        push!(controls, c)

        # Penalize deviation from the sphere surface
        # dist_penalty = (distance_from_center - radius)^2
        d = norm(c - origin)
        radial_penalty += (d - radius)^2
    end

    # Integration Loop
    total_val = 0.0
    total_len = 0.0

    dt = 1.0 / (samples - 1)
    prev_pos = p1

    for i in 1:samples
        t = (i - 1) * dt
        pos = eval_projected_spline(p1, p2, controls, t, origin, radius)

        val = sample_field(sampler, pos)

        step_len = norm(pos - prev_pos)
        total_len += step_len

        # Weighted Path Integral
        total_val += val * step_len

        prev_pos = pos
    end

    # Combined Regularization
    # 1. Length Penalty (w_reg): Keeps path short/straight
    # 2. Radial Penalty (w_radial): Keeps control points on the surface

    final_reg = (w_reg * total_len^2) + (w_radial * radial_penalty)

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
    optimize_spline_connector(p1, p2, mesh, sampler; type=:valley, threshold=0.01)

Finds the optimal path between p1 and p2. 
Iteratively tests Order 0, 1, and 2 splines. 
Accepts higher order only if cost improves by `threshold` %.
"""
function optimize_spline_connector(
    p1::SVector{3,Float64},
    p2::SVector{3,Float64},
    mesh::SphericalScalarMesh,
    sampler::FieldSampler;
    type=:valley,
    threshold=0.005,
) # Lowered threshold slightly to see if it helps
    origin = mesh.origin
    radius = mesh.radius

    # Debug Header
    # Calculate Euclidean distance to see if points are weirdly close/far
    # dist_chord = norm(p1 - p2)
    # println("\n--- Opt Connector ($type) ---")
    # println(
    #     "Endpoints: $(round.(p1, digits=3)) -> $(round.(p2, digits=3)) (Chord: $(round(dist_chord, digits=3)))",
    # )

    # --- Level 0: Geodesic ---
    cost_0 = path_integral_objective(
        SVector{3,Float64}[],
        p1,
        p2,
        sampler,
        origin,
        radius,
        type;
        w_reg=0.1,
        w_radial=10.0,
    )
    best_path = SVector{3,Float64}[]
    best_cost = cost_0
    best_order = 0

    # println("Order 0 Cost: $(round(cost_0, digits=6))")

    # Optimization Wrapper
    function run_opt(init_controls)
        # Check if init controls are valid
        if any(isnan, reduce(vcat, init_controls))
            println("WARNING: Initial controls contain NaN!")
            return Inf, init_controls
        end

        func =
            c -> path_integral_objective(
                c, p1, p2, sampler, origin, radius, type; w_reg=0.1, w_radial=10.0
            )
        flat_init = reduce(vcat, [Vector(c) for c in init_controls])

        res = optimize(func, flat_init, NelderMead(), Optim.Options(; iterations=150)) # Increased iters

        min_cost = Optim.minimum(res)
        best_flat = Optim.minimizer(res)
        n = length(best_flat) ÷ 3
        optimized_ctrls = [SVector{3}(best_flat[(3 * i - 2):(3 * i)]) for i in 1:n]

        return min_cost, optimized_ctrls
    end

    # --- Level 1: Quadratic ---
    mid = origin + normalize((p1-origin) + (p2-origin)) * radius * 1.0
    cost_1, ctrls_1 = run_opt([mid])

    # Improvement Check
    imp_1 = (best_cost - cost_1) / (abs(best_cost) + 1e-9)
    # println(
    #     "Order 1 Cost: $(round(cost_1, digits=6)) | Imp: $(round(imp_1*100, digits=2))% | C1 dist: $(round(norm(ctrls_1[1]-origin), digits=3))",
    # )

    if imp_1 > threshold
        best_cost = cost_1
        best_path = ctrls_1
        best_order = 1

        # --- Level 2: Cubic ---
        v1 = p1 - origin;
        v2 = p2 - origin
        c1_guess = origin + normalize(0.66*v1 + 0.33*v2) * radius * 1.0
        c2_guess = origin + normalize(0.33*v1 + 0.66*v2) * radius * 1.0

        cost_2, ctrls_2 = run_opt([c1_guess, c2_guess])
        imp_2 = (best_cost - cost_2) / (abs(best_cost) + 1e-9)

        # println(
        #     "Order 2 Cost: $(round(cost_2, digits=6)) | Imp: $(round(imp_2*100, digits=2))% | C1 dist: $(round(norm(ctrls_2[1]-origin), digits=3))",
        # )

        if imp_2 > threshold
            best_cost = cost_2
            best_path = ctrls_2
            best_order = 2
        end
    end

    # println("SELECTED: Order $best_order")

    # Generate Samples with simple linear T (This creates the spacing issue if C is far)
    final_points = SVector{3,Float64}[]
    samples = 40 # Increased samples
    for i in 1:samples
        t = (i-1)/(samples-1)
        push!(final_points, eval_projected_spline(p1, p2, best_path, t, origin, radius))
    end

    return final_points
end

"""
    find_spline_intersection(path1, path2, origin, radius)

Finds the crossing point of two discrete paths on a sphere.
Returns the SVector coordinate of the intersection.
"""
function find_spline_intersection(
    path1::Vector{SVector{3,Float64}},
    path2::Vector{SVector{3,Float64}},
    origin::SVector{3,Float64},
    radius::Float64,
)
    min_dist_sq = Inf
    best_p = origin

    # Brute force search (Arrays are small, ~40-50 points, so O(N^2) is negligible)
    for p1 in path1
        for p2 in path2
            # Euclidean distance squared is fine for finding the closest pair on a sphere
            dist_sq = sum(abs2, p1 - p2)

            if dist_sq < min_dist_sq
                min_dist_sq = dist_sq
                # The true crossing is roughly halfway between the closest discrete points
                best_p = p1 + p2
            end
        end
    end

    # Re-project the average point to the exact sphere surface
    return origin + normalize(best_p - 2.0 * origin) * radius
end

"""
    trace_chemical_skeleton(topo::ChemicalTopology, mesh::SphericalScalarMesh, sampler::FieldSampler)

Generates the smooth 1-skeleton by connecting the relaxed critical points.
Returns a dictionary of spline paths.
"""
function trace_chemical_skeleton(
    topo::ChemicalTopology, mesh::SphericalScalarMesh, sampler::FieldSampler
)
    paths = Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}()

    # 1. Connect Minima to Saddles (Valleys)
    # We need to know WHICH Min connects to WHICH Saddle.
    # In the Voronoi model, a Saddle (Edge Midpoint) connects to the two Minima (Vertices) 
    # that define that edge.
    # We need to recover this adjacency. 
    # Fortunately, `predict_topology_voronoi` logic implies:
    # Saddle[k] lies on Edge[k]. Edge[k] connects Max[i] and Max[j].
    # Wait, the Voronoi Edge connects MINIMA. The Delaunay Edge connects MAXIMA.

    # Let's rebuild the adjacency explicitly using Nearest Neighbors or the Dual Graph.
    # Robust method: A Saddle is physically located between two Maxima and two Minima.
    # We can find them by distance or gradient flow. 

    # For robust connectivity without re-running the geometric code:
    # 1. For each Saddle, find the 2 closest Maxima and 2 closest Minima.
    # 2. Connect them.

    n_sad = length(topo.saddles)

    println("Tracing Skeleton for $n_sad saddles...")

    for s_idx in 1:n_sad
        pos_s = topo.saddles[s_idx]

        # --- Find Neighbors (Geometric Heuristic) ---
        # Find 2 closest Maxima (The "Ridge" Parents)
        dists_max = [great_circle_distance(p, pos_s, mesh.origin) for p in topo.maxima]
        max_indices = partialsortperm(dists_max, 1:2)

        # Find 2 closest Minima (The "Valley" Parents)
        dists_min = [great_circle_distance(p, pos_s, mesh.origin) for p in topo.minima]
        min_indices = partialsortperm(dists_min, 1:2)

        # --- Optimize Paths ---

        # 1. Ridge Paths (Red): Max -> Saddle
        for m_idx in max_indices
            path = optimize_spline_connector(
                topo.maxima[m_idx],
                pos_s,
                mesh,
                sampler;
                type=:ridge, # Maximize path integral
            )
            paths[(:max_sad, m_idx, s_idx)] = path
        end

        # 2. Valley Paths (Blue): Min -> Saddle
        for m_idx in min_indices
            path = optimize_spline_connector(
                topo.minima[m_idx],
                pos_s,
                mesh,
                sampler;
                type=:valley, # Minimize path integral
            )
            paths[(:min_sad, m_idx, s_idx)] = path
        end

        # 3. Slope Paths (Grey): Min -> Max (Direct)
        # Optional: Connect the neighbors directly to visualize the bundle axis
        for mx in max_indices, mn in min_indices
            # Only connect if they share this saddle? Yes.
            path = optimize_spline_connector(
                topo.maxima[mx],
                topo.minima[mn],
                mesh,
                sampler;
                type=:descent, # Gradient descent style
            )
            paths[(:min_max, mn, mx)] = path
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
        # Return empty arrays here.
        return (SVector{3,Float64}[], SVector{3,Float64}[])
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
            ;
            continue;
        end # Collinear points

        dir = plane_normal / norm_val

        # Orient the normal OUTWARD (assuming convex hull property on sphere)
        # Check against one of the points (relative to origin)
        if dot(dir, p1 - origin) < 0
            dir = -dir
        end

        # Check Empty Circumcircle Property
        # This is equivalent to checking if all other points are "below" the plane 
        # defined by (p1, p2, p3).
        # Plane eq: (x - p1) . dir = 0

        is_delaunay = true

        for m in 1:n
            if m == i || m == j || m == k
                ;
                continue;
            end

            # Distance from point m to the plane
            # If dist > epsilon, point is "above" plane -> inside circumcircle
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

    # 2. Convert to Voronoi Features

    n = length(maxima)
    if n < 3
        ;
        return (SVector{3,Float64}[], SVector{3,Float64}[], SVector{2,Int}[]);
    end

    # Predicted Minima = Circumcenters of Delaunay Triangles
    pred_minima = SVector{3,Float64}[]

    # Map to track edges for saddles
    # Key: Sorted Pair (idx1, idx2) -> Value: Count (or reference)
    delaunay_edges = Set{SVector{2,Int}}()

    for tri in delaunay_triplets
        # Re-calculate circumcenter (robustly)
        p1, p2, p3 = maxima[tri[1]], maxima[tri[2]], maxima[tri[3]]
        normal = normalize(cross(p2 - p1, p3 - p1))
        if dot(normal, p1 - origin) < 0
            ;
            normal = -normal;
        end

        push!(pred_minima, origin + normal * radius)

        # Collect Edges for Saddle prediction
        push!(delaunay_edges, SVector{2,Int}(minmax(tri[1], tri[2])))
        push!(delaunay_edges, SVector{2,Int}(minmax(tri[2], tri[3])))
        push!(delaunay_edges, SVector{2,Int}(minmax(tri[3], tri[1])))
    end

    # Predicted Saddles = Midpoints of Delaunay Edges
    # (Geometrically, the Voronoi edge crosses the Delaunay edge. 
    # Ideally, the saddle is the intersection, but the midpoint is a safe, robust guess).
    pred_saddles = SVector{3,Float64}[]

    for edge in delaunay_edges
        p1 = maxima[edge[1]]
        p2 = maxima[edge[2]]

        # Geodesic midpoint
        mid = normalize((p1 - origin) + (p2 - origin))
        push!(pred_saddles, origin + mid * radius)
    end

    return pred_minima, collect(delaunay_edges)
end

"""
    clean_degenerate_topology(minima, saddles, radius; tol_factor=0.05)

Merges features that have relaxed into the same physical location due to 
Delaunay degeneracies (e.g., 4-way symmetric faces).
"""
function clean_degenerate_topology(
    minima::Vector{SVector{3,Float64}},
    saddles::Vector{SVector{3,Float64}},
    origin::SVector{3,Float64},
    radius::Float64;
    tol_factor=0.05,
)
    dist_tol = tol_factor * radius

    # 1. Merge Co-located Minima
    merged_minima = SVector{3,Float64}[]
    used_mins = Set{Int}()

    for i in 1:length(minima)
        if i in used_mins
            ;
            continue;
        end

        # Start a cluster
        cluster = [minima[i]]
        push!(used_mins, i)

        # Find all other minima close to this one
        for j in (i + 1):length(minima)
            if j in used_mins
                ;
                continue;
            end

            if norm(minima[i] - minima[j]) < dist_tol
                push!(cluster, minima[j])
                push!(used_mins, j)
            end
        end

        # Average the cluster to get the true center
        avg_pos = sum(cluster) / length(cluster)

        # CORRECT PROJECTION: Relative to the sphere center
        v_shifted = avg_pos - origin
        p_final = origin + normalize(v_shifted) * radius

        push!(merged_minima, p_final)
    end

    # 2. Annihilate Spurious Saddles
    # If a saddle is at the same location as a minimum, it's a fake diagonal.
    clean_saddles = SVector{3,Float64}[]

    for s in saddles
        is_spurious = false
        for m in merged_minima
            if norm(s - m) < dist_tol
                is_spurious = true
                break
            end
        end

        if !is_spurious
            push!(clean_saddles, s)
        end
    end

    println(
        "Topology Cleanup: Merged $(length(minima)) minima -> $(length(merged_minima)). Removed $(length(saddles) - length(clean_saddles)) spurious saddles.",
    )

    return merged_minima, clean_saddles
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
    check_poincare_hopf(grad::DiscreteGradient; verbose=true)

Validates the Euler Characteristic constraint for a sphere (χ=2).
Returns true if valid.
"""
function check_poincare_hopf(grad::DiscreteGradient; verbose=true)
    n_min = length(grad.critical_vertices)
    n_sad = length(grad.critical_edges)
    n_max = length(grad.critical_triangles)

    euler_char = n_min - n_sad + n_max

    if verbose
        println("--- Poincaré-Hopf Check ---")
        println("Minima:  $n_min")
        println("Saddles: $n_sad")
        println("Maxima:  $n_max")
        println("Result:  $n_min - $n_sad + $n_max = $euler_char")
        println("Expected: 2")
    end

    return euler_char == 2
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

struct MorseSkeleton
    # A path is a Vector of Node Indices (integers)
    # We store them by the Saddle that generates them
    saddle_to_min::Dict{Int,Vector{Vector{Int}}} # SaddleIdx -> [Path1, Path2]
    saddle_to_max::Dict{Int,Vector{Vector{Int}}} # SaddleIdx -> [Path1, Path2]

    # Adjacency Graph (optional, for finding Min-Max neighbors)
    adjacency::Dict{Int,Set{Int}} # CriticalPointIdx -> Set of Connected CriticalPointIndices
end

"""
    extract_skeleton(grad, mesh)

Traces all ascending and descending manifolds from every critical saddle.
Returns the 1-Skeleton graph.
"""
function extract_skeleton(grad::DiscreteGradient, mesh::SphericalScalarMesh)
    # We need the Edge->Triangle map for ascending traces
    # (Rebuild or pass from simplify step. Rebuilding here for safety.)
    n_e = length(mesh.edges)
    edge_to_triangles = [Int[] for _ in 1:n_e]
    edge_lookup = Dict(e => i for (i, e) in enumerate(mesh.edges))
    for (t_idx, tri) in enumerate(mesh.triangles)
        for pair in [(tri[1], tri[2]), (tri[2], tri[3]), (tri[3], tri[1])]
            s_pair = SVector{2,Int}(minmax(pair[1], pair[2]))
            push!(edge_to_triangles[edge_lookup[s_pair]], t_idx)
        end
    end

    skel_min = Dict{Int,Vector{Vector{Int}}}()
    skel_max = Dict{Int,Vector{Vector{Int}}}()

    # Iterate all SURVIVING saddles
    for s_idx in grad.critical_edges
        # 1. Trace Down (Saddle -> Min)
        mins, min_paths = trace_descending_path(grad, mesh, s_idx)

        # Convert topological paths (v,e,v,e) to just Vertex geometry for plotting
        # trace_descending returns [v, e, v...]. We just want 'v'.
        geom_min_paths = Vector{Int}[]
        for p in min_paths
            # Filter for vertices (odd indices in the path array)
            # path structure: v0, e1, v1, e2... 
            # Actually, check your trace_descending implementation.
            # My previous impl: push(v), push(e). So stride is 2.
            push!(geom_min_paths, p[1:2:end])
        end
        skel_min[s_idx] = geom_min_paths

        # 2. Trace Up (Saddle -> Max)
        maxs, max_paths = trace_ascending_path(grad, mesh, s_idx, edge_to_triangles)

        geom_max_paths = Vector{Int}[]
        for p in max_paths
            # trace_ascending returns [e, t, e, t...]. 
            # We need to convert this to Vertex positions.
            # A simple approximation: Use the centroid of the Triangle t, 
            # and the midpoint of Edge e.
            # OR, for continuity, we construct a vertex path:
            # Midpoint(e0) -> Centroid(t0) -> Midpoint(e1)...

            # Since plotting usually requires lines between nodes, 
            # let's store the raw element indices and convert to coords later.
            # The path is mixed (Edge/Triangle).
            push!(geom_max_paths, p)
        end
        skel_max[s_idx] = geom_max_paths
    end

    return MorseSkeleton(skel_min, skel_max, Dict{Int,Set{Int}}())
end

"""
    smooth_path_on_sphere(mesh, path_indices, type_path; iterations=3)

Smooths a zig-zag path of indices into a smooth curve of 3D coordinates.
type_path: :vertex (for descending) or :mixed (for ascending)
"""
function smooth_path_on_sphere(
    mesh::SphericalScalarMesh, path::Vector{Int}, type_path::Symbol; iterations=3
)
    # 1. Convert indices to initial 3D coordinates
    coords = SVector{3,Float64}[]

    if type_path == :vertex
        for idx in path
            push!(coords, mesh.positions[idx])
        end
    elseif type_path == :mixed
        # Ascending path: [Edge, Tri, Edge, Tri...]
        for (i, idx) in enumerate(path)
            if isodd(i) # Edge
                v1, v2 = mesh.edges[idx]
                p = (mesh.positions[v1] + mesh.positions[v2]) / 2.0
                push!(coords, p)
            else # Triangle
                v1, v2, v3 = mesh.triangles[idx]
                p = (mesh.positions[v1] + mesh.positions[v2] + mesh.positions[v3]) / 3.0
                push!(coords, p)
            end
        end
    end

    if length(coords) < 3
        return coords
    end

    # 2. Iterative Smoothing (Laplacian with constraints)
    # Keep endpoints fixed (they are the critical points!)
    fixed_start = coords[1]
    fixed_end = coords[end]

    smoothed = copy(coords)

    for _ in 1:iterations
        new_coords = copy(smoothed)
        for i in 2:(length(smoothed) - 1)
            # Simple average of neighbors
            # p_new = 0.25*prev + 0.5*curr + 0.25*next
            p_avg = 0.25 * smoothed[i - 1] + 0.5 * smoothed[i] + 0.25 * smoothed[i + 1]

            # Project back to sphere surface
            # vector from origin
            v = p_avg - mesh.origin
            dist = norm(v)
            if dist > 1e-9
                # Renormalize to radius
                p_projected = mesh.origin + (v / dist) * mesh.radius
                new_coords[i] = p_projected
            else
                new_coords[i] = p_avg
            end
        end
        smoothed = new_coords
    end

    return smoothed
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

"""
    plot_results(mesh::SphericalScalarMesh, grad::DiscreteGradient; 
                 skeleton::Union{MorseSkeleton, Nothing}=nothing,
                 smooth_iters::Int=5)

Visualizes the scalar field, critical points, and optionally the 1-skeleton (separatrices).
- **Surface**: Colored by charge density.
- **Spheres**: Minima (Blue), Saddles (Green), Maxima (Red).
- **Lines** (if skeleton provided): 
    - **Blue Lines**: Descending manifolds (Saddle → Min).
    - **Red Lines**: Ascending manifolds (Saddle → Max).
"""
function plot_results(
    mesh::SphericalScalarMesh,
    grad::DiscreteGradient;
    skeleton::Union{MorseSkeleton,Nothing}=nothing,
    smooth_iters::Int=10,
    plot_title="",
)
    traces = GenericTrace[]

    # --- 1. Base Mesh Surface ---
    # Convert SVectors to simple arrays
    x = [p[1] for p in mesh.positions]
    y = [p[2] for p in mesh.positions]
    z = [p[3] for p in mesh.positions]

    # 0-based indexing for Plotly
    i_idx = [t[1]-1 for t in mesh.triangles]
    j_idx = [t[2]-1 for t in mesh.triangles]
    k_idx = [t[3]-1 for t in mesh.triangles]

    push!(
        traces,
        mesh3d(;
            x=x,
            y=y,
            z=z,
            i=i_idx,
            j=j_idx,
            k=k_idx,
            intensity=mesh.values,
            colorscale="Viridis",
            opacity=1.0, # Lower opacity to see the skeleton inside/on top
            name="Density",
            showscale=true,
        ),
    )

    # --- 2. Critical Points ---
    # Helper to extract coords for specific indices
    function get_cp_coords(indices, type_offset)
        px, py, pz = Float64[], Float64[], Float64[]
        for idx in indices
            if type_offset == 0 # Vertex (Min)
                p = mesh.positions[idx]
            elseif type_offset == 1 # Edge (Saddle)
                v1, v2 = mesh.edges[idx]
                p = (mesh.positions[v1] + mesh.positions[v2]) / 2
            elseif type_offset == 2 # Triangle (Max)
                v1, v2, v3 = mesh.triangles[idx]
                p = (mesh.positions[v1] + mesh.positions[v2] + mesh.positions[v3]) / 3
            end
            push!(px, p[1]);
            push!(py, p[2]);
            push!(pz, p[3])
        end
        return px, py, pz
    end

    # Minima (Blue)
    mx, my, mz = get_cp_coords(grad.critical_vertices, 0)
    push!(
        traces,
        scatter3d(;
            x=mx,
            y=my,
            z=mz,
            mode="markers",
            marker=attr(; size=5, color="blue", symbol="circle"),
            name="Minima",
        ),
    )

    # Saddles (Green)
    sx, sy, sz = get_cp_coords(grad.critical_edges, 1)
    push!(
        traces,
        scatter3d(;
            x=sx,
            y=sy,
            z=sz,
            mode="markers",
            marker=attr(; size=4, color="green", symbol="diamond"),
            name="Saddles",
        ),
    )

    # Maxima (Red)
    Xx, Xy, Xz = get_cp_coords(grad.critical_triangles, 2)
    push!(
        traces,
        scatter3d(;
            x=Xx,
            y=Xy,
            z=Xz,
            mode="markers",
            marker=attr(; size=5, color="red", symbol="circle"),
            name="Maxima",
        ),
    )

    # --- 3. The 1-Skeleton (Optional) ---
    if skeleton !== nothing
        println("Generating Skeleton Traces...")

        # We use NaN separators to draw all lines of one type in a single trace
        # This is strictly for performance (Plotly chokes on 1000s of trace objects)

        function build_lines(saddle_map, type_sym, color_str)
            lx, ly, lz = Float64[], Float64[], Float64[]

            for (s_idx, paths_list) in saddle_map
                for raw_path in paths_list
                    # Smooth the path
                    curve = smooth_path_on_sphere(
                        mesh, raw_path, type_sym; iterations=smooth_iters
                    )

                    # Append points
                    for p in curve
                        push!(lx, p[1]);
                        push!(ly, p[2]);
                        push!(lz, p[3])
                    end
                    # Append NaN to break the line before the next path
                    push!(lx, NaN);
                    push!(ly, NaN);
                    push!(lz, NaN)
                end
            end

            return scatter3d(;
                x=lx,
                y=ly,
                z=lz,
                mode="lines",
                line=attr(; color=color_str, width=4),
                name=type_sym == :vertex ? "Descending (Min-Sad)" : "Ascending (Max-Sad)",
            )
        end

        # Descending (Saddle -> Min) -> Blue Lines
        if !isempty(skeleton.saddle_to_min)
            push!(traces, build_lines(skeleton.saddle_to_min, :vertex, "royalblue"))
        end

        # Ascending (Saddle -> Max) -> Red Lines
        if !isempty(skeleton.saddle_to_max)
            push!(traces, build_lines(skeleton.saddle_to_max, :mixed, "firebrick"))
        end
    end

    # --- 4. Layout & Render ---
    layout = Layout(;
        title=plot_title,
        scene=attr(;
            aspectmode="data",
            xaxis=attr(; visible=false),
            yaxis=attr(; visible=false),
            zaxis=attr(; visible=false),
        ),
        showlegend=true,
    )

    plot(traces, layout)
end

"""
    plot_refined_topology(mesh, topo::ChemicalTopology; title="Refined Topology")

Plots the continuous critical points (Max/Min/Sad) over the mesh surface.
"""
function plot_refined_topology(
    mesh::SphericalScalarMesh, topo::ChemicalTopology; title="Refined Topology"
)

    # 1. Surface
    x = [p[1] for p in mesh.positions]
    y = [p[2] for p in mesh.positions]
    z = [p[3] for p in mesh.positions]
    i_idx = [t[1]-1 for t in mesh.triangles]
    j_idx = [t[2]-1 for t in mesh.triangles]
    k_idx = [t[3]-1 for t in mesh.triangles]

    surf = mesh3d(;
        x=x,
        y=y,
        z=z,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        intensity=mesh.values,
        colorscale="Viridis",
        opacity=1.0,
        name="Density",
    )

    # 2. Critical Points (from SVector list)
    function get_trace(coords, color, name, symbol)
        if isempty(coords)
            return scatter3d(; x=[], y=[], z=[])
        end
        point_lift = 0.01 * mesh.radius
        lifted_coords = SVector{3,Float64}[]
        for p in coords
            radial = p - mesh.origin
            radial_norm = norm(radial)
            if radial_norm > 1e-12
                push!(
                    lifted_coords,
                    mesh.origin + radial * ((mesh.radius + point_lift) / radial_norm),
                )
            else
                push!(lifted_coords, p)
            end
        end

        cx = [p[1] for p in lifted_coords]
        cy = [p[2] for p in lifted_coords]
        cz = [p[3] for p in lifted_coords]
        return scatter3d(;
            x=cx,
            y=cy,
            z=cz,
            mode="markers",
            marker=attr(; size=5, color=color, symbol=symbol),
            name=name,
        )
    end

    t_max = get_trace(topo.maxima, "red", "Refined Max", "circle")
    t_min = get_trace(topo.minima, "blue", "Refined Min", "circle")
    t_sad = get_trace(topo.saddles, "green", "Refined Sad", "diamond")

    layout = Layout(; title=title, scene=attr(; aspectmode="data"), showlegend=true)

    plot([surf, t_max, t_min, t_sad], layout)
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
            showscale=true,
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

    add_paths(:max_sad, "red", "Ridges", 5)
    add_paths(:min_sad, "royalblue", "Valleys", 5)
    add_paths(:min_max, "grey", "Slope", 2) # Thinner, subtle

    # # 4. Connector Points (all connector vertices)
    # cx, cy, cz = Float64[], Float64[], Float64[]
    # for (_, path) in topo.connectors
    #     for p in path
    #         push!(cx, p[1]);
    #         push!(cy, p[2]);
    #         push!(cz, p[3])
    #     end
    # end
    # if !isempty(cx)
    #     push!(
    #         traces,
    #         scatter3d(;
    #             x=cx,
    #             y=cy,
    #             z=cz,
    #             mode="markers",
    #             marker=attr(; size=2, color="black"),
    #             name="Connector Points",
    #         ),
    #     )
    # end

    layout = Layout(; title=title, scene=attr(; aspectmode="data"), showlegend=true)
    plot(traces, layout)
end

function process_sphere(input_file_path::String)
    println("Deserializing SphericalScalarMesh from: $input_file_path")
    spherical_mesh = deserialize(open(input_file_path, "r"))

    input_file_basename = splitext(basename(input_file_path))[1]

    range = maximum(spherical_mesh.values) - minimum(spherical_mesh.values)

    # 1. Compute Gradient of smoothed mesh
    smoothed_mesh = anisotropic_smooth(
        spherical_mesh; iterations=20, lambda=0.05, k=0.02*range
    )
    smoothed_mesh = smooth_mesh_hybrid(
        smoothed_mesh; median_iters=1, laplacian_iters=5, lambda=0.5
    )
    smoothed_discrete_grad = compute_discrete_gradient(smoothed_mesh)

    # Print min/max/mean/stddev of the smoothed values for debugging
    println("\n--- Smoothed Mesh Statistics ---")
    println("Min Value: $(minimum(smoothed_mesh.values))")
    println("Max Value: $(maximum(smoothed_mesh.values))")
    println("Mean Value: $(mean(smoothed_mesh.values))")
    println("Std Dev: $(std(smoothed_mesh.values))")
    println("--------------------------------\n")

    # 2. Auto-Detect Threshold
    thresh = suggest_persistence_threshold(smoothed_mesh, smoothed_discrete_grad; plot=true)

    # 3. Simplify to find Anchors (Maxima)
    simplify_persistence!(smoothed_discrete_grad, smoothed_mesh, thresh)

    # 1. Identify and Refine Anchors
    # This now returns high-precision SVector{3} coordinates
    primary_anchors = identify_primary_anchors(
        smoothed_mesh, smoothed_discrete_grad; min_drop_ratio=2.0
    )

    # 2. Predict Topology
    if length(primary_anchors) >= 3
        println("Regime: Molecular. Predicting topology...")

        # # Voronoi generator uses the REFINED anchors
        # pred_min, pred_sad = predict_topology_voronoi(
        #     primary_anchors, 
        #     smoothed_mesh.origin, 
        #     smoothed_mesh.radius
        # )

        # # 3. Create Topology Object for Plotting/Next Steps
        # # We use primary_anchors for maxima (they are already refined!)
        # pred_topo = ChemicalTopology(
        #     primary_anchors, # Maxima
        #     pred_min,        # Minima (Predicted)
        #     pred_sad,        # Saddles (Predicted)
        #     Dict{Tuple{Symbol, Int, Int}, Vector{SVector{3, Float64}}}()
        # )

        # # 4. Visualize
        # plot_refined_topology(smoothed_mesh, pred_topo, title="Refined Anchors & Predicted Topology")

        # 1. Build Sampler
        sampler = build_sampler(smoothed_mesh)

        # 2. Get Predicted Points AND Connectivity
        # (Note: Update your predict function to return the edges vector!)
        pred_min, pred_edges = predict_topology_voronoi(
            primary_anchors, smoothed_mesh.origin, smoothed_mesh.radius
        )

        # 3. Relax Minima (Elastic)
        refined_min = relax_minima_constrained(
            pred_min, sampler, smoothed_mesh.origin, smoothed_mesh.radius; stiffness=50.0
        )

        # 4. Relax Saddles (Geodesic)
        refined_sad = relax_saddles_geodesic(
            primary_anchors,
            pred_edges,
            sampler,
            smoothed_mesh.origin,
            smoothed_mesh.radius;
            stiffness=50.0,
        )

        # 5. Visualize
        final_topo = ChemicalTopology(
            primary_anchors,
            refined_min,
            refined_sad,
            Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}(),
        )
        plot_refined_topology(smoothed_mesh, final_topo; title="Fully Relaxed Topology")

    else
        println("Regime: Linear/Terminal (N=$(length(primary_anchors))).")
        # Handle N=1 or N=2 logic here (Antipodal / Manifold)
    end
end

"""
    analyze_condensed_density(mesh::SphericalScalarMesh; 
                              smooth_iters=5, 
                              persistence_factor=1.5)

Full Topological Analysis Pipeline:
1. Smoothing (Hybrid Median/Laplacian)
2. Anchor Identification (Persistence & Prominence)
3. Geometric Prediction (Voronoi/Antipodal)
4. Elastic Relaxation (Constrained Optimization)
5. Skeleton Tracing (Spline-NEB)
"""
function analyze_condensed_density(
    mesh::SphericalScalarMesh; smooth_iters=10, persistence_factor=2.0
)
    println("--- 1. Smoothing Mesh ---")
    # 1. Smooth (Hybrid: 1 Median to kill spikes, N Laplacians to smooth basins)
    range = maximum(mesh.values) - minimum(mesh.values)

    # 1. Compute Gradient of smoothed mesh
    smooth_mesh = anisotropic_smooth(mesh; iterations=20, lambda=0.05, k=0.02*range)
    smooth_mesh = smooth_mesh_hybrid(
        smooth_mesh; median_iters=1, laplacian_iters=smooth_iters
    )
    sampler = build_sampler(smooth_mesh)

    println("--- 2. Discrete Topology Analysis ---")
    # 2. Compute Gradient & Threshold
    grad = compute_discrete_gradient(smooth_mesh)
    p_thresh = suggest_persistence_threshold(smooth_mesh, grad)
    simplify_persistence!(grad, smooth_mesh, p_thresh)

    # 3. Identify Anchors (Refined Maxima)
    anchors = identify_primary_anchors(smooth_mesh, grad; min_drop_ratio=persistence_factor)
    n_anchors = length(anchors)
    println("Identified $n_anchors primary anchors.")

    # 4. Regime Detection & Prediction
    pred_min, pred_sad, pred_edges = SVector{3,Float64}[],
    SVector{3,Float64}[],
    SVector{2,Int}[]

    if n_anchors >= 3
        println("Regime: Molecular (Voronoi Prediction)")
        pred_min, pred_edges = predict_topology_voronoi(
            anchors, smooth_mesh.origin, smooth_mesh.radius
        )
        # Convert edges to saddles (midpoints)
        for e in pred_edges
            mid = normalize(
                (anchors[e[1]] - smooth_mesh.origin) + (anchors[e[2]] - smooth_mesh.origin)
            )
            push!(pred_sad, smooth_mesh.origin + mid * smooth_mesh.radius)
        end

    elseif n_anchors == 1
        println("Regime: Terminal (Antipodal Prediction)")
        # Predict minimum at antipode
        v_max = anchors[1] - smooth_mesh.origin
        push!(pred_min, smooth_mesh.origin - v_max) # Antipode
    # No saddles in N=1

    elseif n_anchors == 2
        println("Regime: Linear/Diatomic (Manifold Analysis)")
        # (Placeholder for the 1D Manifold logic discussed earlier)
        # For now, treat as simple gradient bundle or skip Voronoi
    end

    println("--- 3. Elastic Relaxation ---")
    # 5. Relax Points
    refined_min = relax_minima_constrained(
        pred_min, sampler, smooth_mesh.origin, smooth_mesh.radius; stiffness=1.0
    )
    refined_sad = relax_saddles_geodesic(
        anchors,
        pred_edges,
        sampler,
        smooth_mesh.origin,
        smooth_mesh.radius;
        stiffness=0.001,
    )
    clean_min, clean_sad = clean_degenerate_topology(
        refined_min, refined_sad, smooth_mesh.origin, smooth_mesh.radius; tol_factor=0.15
    )

    # Construct Topology Object
    topo = ChemicalTopology(
        anchors,
        clean_min,
        clean_sad,
        Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}(),
    )

    println("--- 4. Tracing Skeleton ---")
    # 6. Trace Paths
    topo.connectors = trace_chemical_skeleton(topo, smooth_mesh, sampler)

    return topo, smooth_mesh
end

"""
    compare_spherical_mesh_to_tsv(mesh, tsv_path; top_n=20)

Compares a `SphericalScalarMesh` against a TSV file with rows in the form:
`X<TAB>Y<TAB>Z<TAB>value`.

Assumes TSV row ordering should match `mesh.positions` and `mesh.values` ordering.
Prints summary statistics for:
- value differences (`mesh.values - tsv_value`)
- Euclidean position distances (`norm(mesh.positions - tsv_position)`)

Also prints top-`N` outliers by absolute value difference and by position distance.
"""
function compare_spherical_mesh_to_tsv(
    mesh::SphericalScalarMesh, tsv_path::String; top_n::Int=20
)
    if !isfile(tsv_path)
        error("TSV file does not exist: $tsv_path")
    end

    tsv_positions = SVector{3,Float64}[]
    tsv_values = Float64[]

    open(tsv_path, "r") do io
        for (line_number, raw_line) in enumerate(eachline(io))
            line = strip(raw_line)
            if isempty(line) || startswith(line, "#")
                continue
            end

            fields = split(line, '\t')
            if length(fields) != 4
                error(
                    "Invalid TSV format at line $line_number in $tsv_path: expected 4 tab-separated fields (X Y Z value), got $(length(fields))",
                )
            end

            x = parse(Float64, strip(fields[1]))
            y = parse(Float64, strip(fields[2]))
            z = parse(Float64, strip(fields[3]))
            value = parse(Float64, strip(fields[4]))

            push!(tsv_positions, SVector{3,Float64}(x, y, z))
            push!(tsv_values, value)
        end
    end

    mesh_n = length(mesh.values)
    tsv_n = length(tsv_values)
    n_compare = min(mesh_n, tsv_n)

    println("\n=== Mesh vs TSV Comparison ===")
    println("TSV file: $tsv_path")
    println("Mesh nodes: $mesh_n")
    println("TSV rows: $tsv_n")
    println("Compared rows: $n_compare")

    if mesh_n != length(mesh.positions)
        println(
            "[WARN] Mesh positions/value length mismatch: positions=$(length(mesh.positions)), values=$mesh_n",
        )
    end
    if tsv_n != length(tsv_positions)
        println(
            "[WARN] TSV positions/value length mismatch: positions=$(length(tsv_positions)), values=$tsv_n",
        )
    end
    if mesh_n != tsv_n
        println(
            "[WARN] Length mismatch between mesh and TSV; comparison truncated to first $n_compare rows.",
        )
    end

    if n_compare == 0
        println("No rows available for comparison.")
        println("==============================\n")
        return nothing
    end

    value_diffs = Vector{Float64}(undef, n_compare)
    abs_value_diffs = Vector{Float64}(undef, n_compare)
    position_distances = Vector{Float64}(undef, n_compare)

    for i in 1:n_compare
        dval = mesh.values[i] - tsv_values[i]
        pdist = norm(mesh.positions[i] - tsv_positions[i])
        value_diffs[i] = dval
        abs_value_diffs[i] = abs(dval)
        position_distances[i] = pdist
    end

    _safe_std(v) = length(v) > 1 ? std(v) : 0.0

    println("\n--- Value Difference Stats (mesh - tsv) ---")
    println("Min   : $(minimum(value_diffs))")
    println("Max   : $(maximum(value_diffs))")
    println("Mean  : $(mean(value_diffs))")
    println("StdDev: $(_safe_std(value_diffs))")
    println(
        "|Diff| Min/Max/Mean/StdDev: $(minimum(abs_value_diffs)) / $(maximum(abs_value_diffs)) / $(mean(abs_value_diffs)) / $(_safe_std(abs_value_diffs))",
    )

    println("\n--- Position Distance Stats (Euclidean) ---")
    println("Min   : $(minimum(position_distances))")
    println("Max   : $(maximum(position_distances))")
    println("Mean  : $(mean(position_distances))")
    println("StdDev: $(_safe_std(position_distances))")

    n_outliers = min(top_n, n_compare)

    value_outlier_idx = sortperm(abs_value_diffs; rev=true)[1:n_outliers]
    println("\n--- Top $n_outliers Value Outliers (by |mesh - tsv|) ---")
    for rank in 1:n_outliers
        idx = value_outlier_idx[rank]
        println(
            "[$rank] idx=$idx, mesh_value=$(mesh.values[idx]), tsv_value=$(tsv_values[idx]), diff=$(value_diffs[idx]), abs_diff=$(abs_value_diffs[idx])",
        )
    end

    position_outlier_idx = sortperm(position_distances; rev=true)[1:n_outliers]
    println("\n--- Top $n_outliers Position Outliers (by Euclidean distance) ---")
    for rank in 1:n_outliers
        idx = position_outlier_idx[rank]
        println(
            "[$rank] idx=$idx, mesh_pos=$(mesh.positions[idx]), tsv_pos=$(tsv_positions[idx]), dist=$(position_distances[idx])",
        )
    end

    println("================================\n")
    return nothing
end

"""
    get_mesh_node_index_and_value(mesh, xyz; exact_tol=1e-10, verbose=true)

Finds the node in `mesh.positions` corresponding to `xyz` and returns its node index
and scalar value from `mesh.values`.

Behavior:
- If a node exists within `exact_tol` Euclidean distance from `xyz`, returns that node.
- Otherwise returns the nearest node on the mesh.

Returns a named tuple:
`(index, value, position, distance, matched_exactly)`
"""
function get_mesh_node_index_and_value(
    mesh::SphericalScalarMesh,
    xyz::SVector{3,Float64};
    exact_tol::Float64=1e-10,
    verbose::Bool=true,
)
    n_nodes = length(mesh.positions)
    if n_nodes == 0
        error("Mesh has no nodes.")
    end
    if length(mesh.values) != n_nodes
        error(
            "Mesh position/value length mismatch: positions=$n_nodes, values=$(length(mesh.values))",
        )
    end

    nearest_idx = 1
    nearest_dist = norm(mesh.positions[1] - xyz)

    exact_idx = nearest_dist <= exact_tol ? 1 : 0

    for i in 2:n_nodes
        d = norm(mesh.positions[i] - xyz)
        if d < nearest_dist
            nearest_dist = d
            nearest_idx = i
        end
        if exact_idx == 0 && d <= exact_tol
            exact_idx = i
        end
    end

    idx = exact_idx != 0 ? exact_idx : nearest_idx
    dist = norm(mesh.positions[idx] - xyz)
    matched_exactly = dist <= exact_tol
    value = mesh.values[idx]

    result = (
        index=idx,
        value=value,
        position=mesh.positions[idx],
        distance=dist,
        matched_exactly=matched_exactly,
    )

    if verbose
        println("\n--- Mesh Node Lookup ---")
        println("Query XYZ      : $xyz")
        println("Node Index     : $(result.index)")
        println("Node Position  : $(result.position)")
        println("Node Value     : $(result.value)")
        println("Distance       : $(result.distance)")
        println("Matched Exactly: $(result.matched_exactly)")
        println("------------------------\n")
    end

    return result
end

function get_mesh_node_index_and_value(
    mesh::SphericalScalarMesh,
    xyz::AbstractVector{<:Real};
    exact_tol::Float64=1e-10,
    verbose::Bool=true,
)
    if length(xyz) != 3
        error("xyz must have exactly 3 elements [x, y, z]; got length $(length(xyz)).")
    end
    xyz_svec = SVector{3,Float64}(Float64(xyz[1]), Float64(xyz[2]), Float64(xyz[3]))
    return get_mesh_node_index_and_value(
        mesh, xyz_svec; exact_tol=exact_tol, verbose=verbose
    )
end

function process_sphere_new(input_path::String)
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

    return process_sphere_new(input_path)
end
