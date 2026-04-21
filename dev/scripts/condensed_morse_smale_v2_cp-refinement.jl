using ElectronDensityAnalysis: SphericalScalarMesh, great_circle_distance
using Serialization: deserialize
using StaticArrays
using LinearAlgebra
using Statistics
using PlotlyJS

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

struct ChemicalTopology
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

# Tie-breaker helper
function is_less(i, j, values)
    if values[i] != values[j]
        return values[i] < values[j]
    end
    return i < j
end

"""
    predict_topology_voronoi(maxima::Vector{SVector{3, Float64}}, sphere_origin, sphere_radius)

Computes the Spherical Voronoi diagram of the maxima to predict Minima and Saddles.
Returns: (predicted_minima, predicted_saddles)
"""
function predict_topology_voronoi(maxima::Vector{SVector{3,Float64}}, origin, radius)
    n = length(maxima)
    if n < 3
        return (SVector{3,Float64}[], SVector{3,Float64}[])
    end

    delaunay_triplets = SVector{3,Int}[]
    voronoi_vertices = SVector{3,Float64}[]     # These are Candidate Minima
    voronoi_edge_midpoints = SVector{3,Float64}[] # These are Candidate Saddles

    # 1. Brute-force Delaunay (O(N^4)) - Acceptable for N < 50
    for i in 1:n, j in (i + 1):n, k in (j + 1):n
        p1, p2, p3 = maxima[i], maxima[j], maxima[k]

        # Compute circumcenter (normalized vector sum for sphere?)
        # For a sphere, the direction of the circumcenter is simply the normal 
        # to the plane defined by p1, p2, p3.
        # But we must pick the direction that points OUT of the sphere (assuming points are convex-ish)

        normal = cross(p2 - p1, p3 - p1)
        if norm(normal) < 1e-9
            ;
            continue;
        end # Collinear/Degenerate

        center_dir = normalize(normal)

        # Check orientation: The center should be on the same side as the points?
        # Actually, for spherical convex hull, we want the plane that cuts off the sphere cap.
        # We need to check if this cap is empty of other points.

        # Signed distance of other points to the plane defined by (p1, p2, p3)
        # Plane eq: (r - p1) . normal = 0
        # If all other points have dot < 0 (or all > 0), it's a Delaunay face.

        is_delaunay = true
        side_ref = 0

        for m in 1:n
            if m == i || m == j || m == k
                ;
                continue;
            end
            dist = dot(maxima[m] - p1, center_dir)

            if abs(dist) < 1e-7
                ;
                continue;
            end # Co-planar point

            if side_ref == 0
                side_ref = sign(dist)
            elseif sign(dist) != side_ref
                is_delaunay = false
                break
            end
        end

        if is_delaunay
            push!(delaunay_triplets, SVector(i, j, k))

            # The Voronoi Vertex is the projection of the circumcenter to surface
            # Correct direction is determined by the "Empty" side.
            # Usually, for convex hull, the normal points OUT.
            # If points were "below" the plane, normal is "above".

            # Simple heuristic: Voronoi vertex should be "between" the maxima.
            # The circumcenter direction.
            # We assume proper convexity.

            # Correct orientation check:
            # If points are "inside", the normal points "outside".
            if side_ref < 0
                push!(voronoi_vertices, origin + center_dir * radius)
            else
                push!(voronoi_vertices, origin - center_dir * radius)
            end
        end
    end

    # 2. Extract Unique Voronoi Edges (Saddles)
    # A Voronoi edge exists between two Voronoi vertices if their Delaunay triangles share an edge.
    # Alternatively, and simpler: The Dual of a Delaunay Edge is a Voronoi Edge.
    # A Delaunay Edge is a pair (i, j) that belongs to at least two Delaunay triangles.
    # The "Saddle" is the midpoint of the geodesic arc between maxima i and j.

    # Let's collect all edges from the identified triplets
    edges = Set{SVector{2,Int}}()
    for t in delaunay_triplets
        push!(edges, SVector{2,Int}(minmax(t[1], t[2])))
        push!(edges, SVector{2,Int}(minmax(t[2], t[3])))
        push!(edges, SVector{2,Int}(minmax(t[3], t[1])))
    end

    for e in edges
        p1 = maxima[e[1]]
        p2 = maxima[e[2]]
        # Midpoint on sphere
        mid = normalize((p1 - origin) + (p2 - origin))
        push!(voronoi_edge_midpoints, origin + mid * radius)
    end

    return voronoi_vertices, voronoi_edge_midpoints
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

function process_sphere(input_file_path::String)
    println("Deserializing SphericalScalarMesh from: $input_file_path")
    spherical_mesh = deserialize(open(input_file_path, "r"))

    input_file_basename = splitext(basename(input_file_path))[1]

    discrete_grad = compute_discrete_gradient(spherical_mesh)
    skeleton = extract_skeleton(discrete_grad, spherical_mesh)

    range = maximum(spherical_mesh.values) - minimum(spherical_mesh.values)

    smoothed_mesh = anisotropic_smooth(
        spherical_mesh; iterations=20, lambda=0.05, k=0.02*range
    )
    smoothed_mesh = smooth_mesh_hybrid(
        smoothed_mesh; median_iters=1, laplacian_iters=5, lambda=0.5
    )
    smoothed_discrete_grad = compute_discrete_gradient(smoothed_mesh)

    # Save mesh stats to variables
    min_val = minimum(smoothed_mesh.values)
    max_val = maximum(smoothed_mesh.values)
    mean_val = mean(smoothed_mesh.values)
    median_val = median(smoothed_mesh.values)
    std_val = std(smoothed_mesh.values)
    range = max_val - min_val

    target_min = 4
    target_sad = 6
    target_max = 4

    # If unsimplified has fewer maxes or mins than target, return.
    if length(smoothed_discrete_grad.critical_vertices) < target_min ||
        length(smoothed_discrete_grad.critical_edges) < target_sad ||
        length(smoothed_discrete_grad.critical_triangles) < target_max
        println(
            "$input_file_basename Smoothed mesh has fewer critical points than target. Skipping simplification.",
        )
        return nothing
    end

    println("$input_file_basename Starting Binary Search for Simplification Threshold...")
    println(
        "Initial counts: minima=$(length(smoothed_discrete_grad.critical_vertices)) saddles=$(length(smoothed_discrete_grad.critical_edges)) maxima=$(length(smoothed_discrete_grad.critical_triangles))",
    )
    println("Target counts: minima=$target_min saddles=$target_sad maxima=$target_max")
    println("Function value range: [$min_val, $max_val] (Range = $range)")

    lower = 0.0
    upper = range

    println("$input_file_basename Initial threshold search range: [$lower, $upper]")

    function evaluate_threshold(thresh::Float64)
        discrete_grad_simplified = deepcopy(smoothed_discrete_grad)
        simplify_persistence!(discrete_grad_simplified, smoothed_mesh, thresh)

        n_min = length(discrete_grad_simplified.critical_vertices)
        n_sad = length(discrete_grad_simplified.critical_edges)
        n_max = length(discrete_grad_simplified.critical_triangles)

        score = abs(n_min - target_min) + abs(n_max - target_max)
        return discrete_grad_simplified, n_min, n_sad, n_max, score
    end

    best_threshold = lower
    best_grad, best_min, best_sad, best_max, best_score = evaluate_threshold(lower)
    exact_match = (
        best_min == target_min && best_sad == target_sad && best_max == target_max
    )

    upper_grad, upper_min, upper_sad, upper_max, upper_score = evaluate_threshold(upper)
    if upper_score < best_score
        best_threshold = upper
        best_grad = upper_grad
        best_min = upper_min
        best_sad = upper_sad
        best_max = upper_max
        best_score = upper_score
        exact_match = (
            best_min == target_min && best_sad == target_sad && best_max == target_max
        )
    end

    max_iters = 20
    for iter in 1:max_iters
        exact_match && break

        mid = (lower + upper) / 2
        mid_grad, n_min, n_sad, n_max, score = evaluate_threshold(mid)
        min_delta = n_min - target_min
        max_delta = n_max - target_max
        extrema_delta = (n_min + n_max) - (target_min + target_max)

        if score < best_score
            best_threshold = mid
            best_grad = mid_grad
            best_min = n_min
            best_sad = n_sad
            best_max = n_max
            best_score = score
        end

        println(
            "$input_file_basename Threshold search iter $iter: Best Score=$best_score Score=$score (Target score = 0) Threshold=$mid | Min=$n_min Saddles=$n_sad Max=$n_max",
        )

        if n_min == target_min && n_sad == target_sad && n_max == target_max
            best_threshold = mid
            best_grad = mid_grad
            best_min = n_min
            best_sad = n_sad
            best_max = n_max
            exact_match = true
            break
        end

        # Use extrema counts to drive bisection direction.
        # - Too many minima/maxima => threshold too low => increase threshold.
        # - Too few minima/maxima  => threshold too high => decrease threshold.
        # If minima/maxima disagree (one too high, one too low), fall back to combined extrema delta.
        too_many_extrema = (min_delta > 0) || (max_delta > 0)
        too_few_extrema = (min_delta < 0) || (max_delta < 0)

        if too_many_extrema && !too_few_extrema
            lower = mid
        elseif too_few_extrema && !too_many_extrema
            upper = mid
        elseif extrema_delta > 0
            lower = mid
        elseif extrema_delta < 0
            upper = mid
        else
            # min/max targets are matched but exact saddle target may differ due numerical/path effects.
            best_threshold = mid
            best_grad = mid_grad
            best_min = n_min
            best_sad = n_sad
            best_max = n_max
            exact_match = true
            break
        end

        if abs(upper - lower) < 1e-12
            break
        end
    end

    refined_topo = refine_critical_points(smoothed_mesh, best_grad)

    if length(refined_topo.maxima) >= 3
        pred_min, pred_sad = predict_topology_voronoi(
            refined_topo.maxima, smoothed_mesh.origin, smoothed_mesh.radius
        )

        # Visual Check: Are the predicted (Voronoi) points close to the refined (Morse) points?
        # This confirms if our chemical prior is valid.

        # We can create a temporary topology to plot predictions
        pred_topo = ChemicalTopology(
            refined_topo.maxima,
            pred_min,
            pred_sad,
            Dict{Tuple{Symbol,Int,Int},Vector{SVector{3,Float64}}}(),
        )

        p1 = plot_refined_topology(smoothed_mesh, refined_topo; title="Morse-Smale Refined")
        p2 = plot_refined_topology(smoothed_mesh, pred_topo; title="Voronoi Predicted")

        display(p1)
        display(p2)
    end
end

function main(
    input_path::String="/Users/haiiro/NoSync/ElectronDensityAnalysis.jl/gbd_checkpoints/ethane_20000/atom_1_ρ_spherical_mesh.jls",
)
    if !isfile(input_path)
        error("File does not exist: $input_path")
    end

    process_sphere(input_path)
end
