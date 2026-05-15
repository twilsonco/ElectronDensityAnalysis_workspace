"""
    SymmetrySpherePoints

Generate uniformly-spaced seed points on a unit sphere that respect a given
crystallographic point-group symmetry.  Points are first placed in the
irreducible wedge (fundamental domain) via a barycentric grid, then expanded
to the full sphere by applying every symmetry operation.

Currently supported groups:
  • `OhSymmetry`  – full octahedral (m-3m), order 48
  • `TdSymmetry`  – tetrahedral (−43m),   order 24

Adding a new group requires implementing three methods:
  `wedge_vertices`, `symmetry_images`, and `group_order`.
"""
module SymmetrySpherePoints

using LinearAlgebra
using Combinatorics   # permutations

export SymmetryGroup, OhSymmetry, TdSymmetry
export wedge_vertices, group_order, symmetry_images
export wedge_points, full_sphere_points
export n_sphere_points, find_q

# ═══════════════════════════════════════════════════════════════════════════════
# Abstract interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    abstract type SymmetryGroup

Supertype for all point-group symmetries.  Concrete subtypes must implement:

  - `wedge_vertices(::G)  → (A, B, C)`   three unit vectors bounding the wedge
  - `group_order(::G)     → Int`          number of symmetry operations
  - `symmetry_images(::G, p) → Vector`    all distinct images of point `p`
"""
abstract type SymmetryGroup end

"""
    wedge_vertices(g::SymmetryGroup) → (A, B, C)

Return the three vertex unit-vectors of the irreducible wedge (spherical
triangle) for symmetry group `g`.
"""
function wedge_vertices end

"""
    group_order(g::SymmetryGroup) → Int

Return the order (number of elements) of symmetry group `g`.
"""
function group_order end

"""
    symmetry_images(g::SymmetryGroup, p::AbstractVector) → Vector{Vector{Float64}}

Return every distinct image of point `p` under the operations of group `g`.
"""
function symmetry_images end

# ═══════════════════════════════════════════════════════════════════════════════
# Generic algorithms (work for any SymmetryGroup)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    wedge_points(g::SymmetryGroup, q::Int; interior_only=false)

Generate seed points in the irreducible wedge of `g` on a `q`-level
barycentric grid.  Returns a vector of `(cart, ang)` tuples where
`cart = [x,y,z]` is a unit vector and `ang = [ϕ,θ]` gives the azimuthal
and polar angles.

If `interior_only=true`, points on the wedge boundary (mirror planes) are
excluded.
"""
function wedge_points(g::SymmetryGroup, q::Int; interior_only::Bool = false)
    A, B, C = wedge_vertices(g)
    pts = Tuple{Vector{Float64}, Vector{Float64}}[]

    i_range = interior_only ? (1:(q - 2)) : (0:q)

    for i in i_range
        j_range = interior_only ? (1:(q - i - 1)) : (0:(q - i))
        for j in j_range
            p  = (i / q) * A + (j / q) * B + (1 - i / q - j / q) * C
            pn = normalize(p)

            ϕ = atan(pn[2], pn[1])
            θ = acos(clamp(pn[3], -1.0, 1.0))

            push!(pts, (pn, [ϕ, θ]))
        end
    end

    return pts
end

"""
    full_sphere_points(g::SymmetryGroup, q::Int; interior_only=false)

Generate the full set of symmetry-expanded points on the unit sphere.
First builds wedge points, then applies all symmetry operations and
deduplicates.

Returns a `Vector{Vector{Float64}}` of Cartesian unit vectors.
"""
function full_sphere_points(g::SymmetryGroup, q::Int; interior_only::Bool = false)
    wpts = wedge_points(g, q; interior_only)
    cart = [p[1] for p in wpts]

    all_pts = Vector{Float64}[]
    for cp in cart
        append!(all_pts, symmetry_images(g, cp))
    end

    # deduplicate via rounding
    return unique(v -> round.(v; digits = 10), all_pts)
end

"""
    n_sphere_points(g::SymmetryGroup, q::Int; interior_only=false) → Int

Return the number of unique full-sphere points for grid level `q` without
materializing the point arrays.  Useful for searching over `q`.
"""
function n_sphere_points(g::SymmetryGroup, q::Int; interior_only::Bool = false)
    return length(full_sphere_points(g, q; interior_only))
end

"""
    find_q(g::SymmetryGroup, n_target::Int; interior_only=false) → (q, n_actual)

Find the grid level `q` whose full-sphere expansion is closest to `n_target`
points.  Returns `(q, n_actual)` where `n_actual` is the true point count
(which may differ from `n_target` due to the discrete grid and merging of
boundary points).

When two values of `q` are equidistant from `n_target`, the larger one
(more points) is chosen.
"""
function find_q(g::SymmetryGroup, n_target::Int; interior_only::Bool = false)
    n_target > 0 || throw(ArgumentError("n_target must be positive"))

    best_q = 1
    best_n = n_sphere_points(g, 1; interior_only)

    for q in 2:10_000
        n = n_sphere_points(g, q; interior_only)
        if abs(n - n_target) <= abs(best_n - n_target)
            best_q, best_n = q, n
        end
        # once we've overshot by more than the current best error, stop
        if n > n_target && (n - n_target) > abs(best_n - n_target)
            break
        end
    end

    return (q = best_q, n = best_n)
end

"""
    full_sphere_points(g::SymmetryGroup; n_target, interior_only=false)

Convenience method: find the best `q` for `n_target` total points, then
return `(points, q, n_actual)`.
"""
function full_sphere_points(g::SymmetryGroup; n_target::Int, interior_only::Bool = false)
    q, n = find_q(g, n_target; interior_only)
    pts  = full_sphere_points(g, q; interior_only)
    return (points = pts, q = q, n = length(pts))
end

"""
    wedge_points(g::SymmetryGroup; n_target, interior_only=false)

Convenience method: find the best `q` for `n_target` total *full-sphere*
points, then return wedge points and metadata.
"""
function wedge_points(g::SymmetryGroup; n_target::Int, interior_only::Bool = false)
    q, n = find_q(g, n_target; interior_only)
    pts  = wedge_points(g, q; interior_only)
    return (points = pts, q = q, n_sphere = n)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Oh  –  full octahedral symmetry  (m-3m),  order 48
# ═══════════════════════════════════════════════════════════════════════════════

"""
    OhSymmetry <: SymmetryGroup

Full octahedral point group (Schoenflies: Oh, Hermann-Mauguin: m-3m).
Order 48.  Operations: all permutations of coordinates × all sign changes.

Irreducible wedge vertices:
  A = [1,0,0]       (C₄ axis)
  B = [1,1,0]/√2    (C₂ axis, edge midpoint)
  C = [1,1,1]/√3    (C₃ axis)
"""
struct OhSymmetry <: SymmetryGroup end

wedge_vertices(::OhSymmetry) = (
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0] / √2,
    [1.0, 1.0, 1.0] / √3,
)

group_order(::OhSymmetry) = 48

function symmetry_images(::OhSymmetry, p::AbstractVector)
    images = Vector{Float64}[]
    for perm in permutations([1, 2, 3])
        for signs in Iterators.product((-1, 1), (-1, 1), (-1, 1))
            img = [s * p[perm[k]] for (k, s) in enumerate(signs)]
            if !any(v -> isapprox(v, img; atol = 1e-12), images)
                push!(images, img)
            end
        end
    end
    return images
end

# ═══════════════════════════════════════════════════════════════════════════════
# Td  –  tetrahedral symmetry  (-43m),  order 24
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TdSymmetry <: SymmetryGroup

Tetrahedral point group (Schoenflies: Td, Hermann-Mauguin: -43m).
Order 24.  Operations: all signed permutations of coordinates having an
even number (0 or 2) of sign flips.

  • 12 proper rotations (T):   even permutations × even sign flips
  • 12 improper operations:    odd permutations  × even sign flips
    (6 S₄ + 6 σd)

Irreducible wedge vertices (Schwarz triangle (2,3,3)):
  A = [1,0,0]         (S₄ / C₂ axis)
  B = [1,1,1]/√3      (C₃ axis)
  C = [1,1,-1]/√3     (C₃ axis)

Wedge edges lie on σd mirror planes: y=z, y=-z, and x=y.
"""
struct TdSymmetry <: SymmetryGroup end

wedge_vertices(::TdSymmetry) = (
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0] / √3,
    [1.0, 1.0, -1.0] / √3,
)

group_order(::TdSymmetry) = 24

function symmetry_images(::TdSymmetry, p::AbstractVector)
    # Td = all signed permutations with an even number of sign flips (0 or 2).
    even_sign_patterns = [
        ( 1,  1,  1),   # 0 negatives
        ( 1, -1, -1),   # 2 negatives
        (-1,  1, -1),   # 2 negatives
        (-1, -1,  1),   # 2 negatives
    ]

    images = Vector{Float64}[]
    for perm in permutations([1, 2, 3])
        for signs in even_sign_patterns
            img = [signs[k] * p[perm[k]] for k in 1:3]
            if !any(v -> isapprox(v, img; atol = 1e-12), images)
                push!(images, img)
            end
        end
    end
    return images
end

end # module
