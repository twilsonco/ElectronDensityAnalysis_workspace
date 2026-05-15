using LinearAlgebra
using Printf
using Combinatorics  # for permutations
using GLMakie

# ── Number of points in the irreducible wedge (interior only) ────────────────
n_wedge_points(q::Int) = (q - 1) * (q - 2) ÷ 2

# Print a summary table
println("  q  │ wedge pts │ full sphere pts")
println("─────┼───────────┼────────────────")
for q in 3:30
    nw = n_wedge_points(q)
    @printf("%4d │ %9d │ %14d\n", q, nw, 48 * nw)
end

# ── Generate points in the irreducible wedge of the Oh point group ───────────
#
# Returns a vector of (cart, ang) tuples where
#   cart = [x, y, z]  (unit vector on the sphere)
#   ang  = [ϕ, θ]     (azimuthal, polar)
#
# This variant includes ALL points (interior + mirror-plane boundary).
# For interior-only, change the loop bounds to i=1:q-2, j=1:q-i-1.

function oh_wedge_points(q::Int)
    A = [1.0, 0.0, 0.0]
    B = [1.0, 1.0, 0.0] / √2
    C = [1.0, 1.0, 1.0] / √3

    pts = Tuple{Vector{Float64}, Vector{Float64}}[]

    for i in 0:q, j in 0:(q - i)
        p  = (i / q) * A + (j / q) * B + (1 - i / q - j / q) * C
        pn = normalize(p)            # cartesian coordinates on the unit sphere

        ϕ = atan(pn[2], pn[1])       # azimuthal angle  (Mathematica's ArcTan[x,y])
        θ = acos(clamp(pn[3], -1, 1)) # polar angle

        push!(pts, (pn, [ϕ, θ]))
    end

    return pts
end

# ── Generate all Oh-symmetry images of a single point ────────────────────────
#
# For a point p = [x, y, z], the Oh group acts by all permutations of
# coordinates combined with all sign changes → 6 × 8 = 48 images (with
# duplicates removed for points on symmetry elements).

function oh_images(p::AbstractVector)
    images = Vector{Float64}[]
    for perm in permutations([1, 2, 3])
        for signs in Iterators.product((-1, 1), (-1, 1), (-1, 1))
            img = [s * p[perm[k]] for (k, s) in enumerate(signs)]
            # deduplicate (use approximate comparison for floating-point)
            if !any(v -> isapprox(v, img; atol = 1e-12), images)
                push!(images, img)
            end
        end
    end
    return images
end

# ── Build wedge + full-sphere points ─────────────────────────────────────────

q = 16
pts      = oh_wedge_points(q)
cart_pts  = [p[1] for p in pts]   # Vector of [x,y,z]
ang_pts   = [p[2] for p in pts]   # Vector of [ϕ,θ]

println("\nWedge points (q=$q): $(length(cart_pts))")

# Expand to full sphere via Oh symmetry
full_sphere_pts_nested = vcat([oh_images(cp) for cp in cart_pts]...)
# Deduplicate
full_sphere_pts = unique(v -> round.(v; digits = 10), full_sphere_pts_nested)

println("Full-sphere points:  $(length(full_sphere_pts))")

# ── Visualization ────────────────────────────────────────────────────────────

function plot_oh_wedge_on_sphere(q::Int)
    pts      = oh_wedge_points(q)
    cart_pts = [p[1] for p in pts]

    fig = Figure(size = (900, 900))
    ax  = Axis3(fig[1, 1];
        aspect = :data,
        xlabel = "x", ylabel = "y", zlabel = "z",
        title  = "Oh irreducible wedge  (q = $q, all points)",
    )

    # translucent sphere
    θs = range(0, π;  length = 60)
    ϕs = range(0, 2π; length = 120)
    xs = [sin(θ) * cos(ϕ) for θ in θs, ϕ in ϕs]
    ys = [sin(θ) * sin(ϕ) for θ in θs, ϕ in ϕs]
    zs = [cos(θ)          for θ in θs, ϕ in ϕs]
    surface!(ax, xs, ys, zs; color = fill((:lightblue, 0.12), size(xs)),
             shading = NoShading, transparency = true)

    # wedge boundary edges
    # Edge 1: great-circle arc from A toward B  (z = 0 plane, t ∈ [0, π/4])
    t1 = range(0, π / 4; length = 200)
    lines!(ax, cos.(t1), sin.(t1), zeros(length(t1)); color = :black, linewidth = 2)

    # Edge 2: A → B
    t2 = range(0, 1; length = 100)
    A = [1.0, 0.0, 0.0]; Bn = [1.0, 1.0, 0.0] / √2
    edge2 = [normalize((1 - t) * A + t * Bn) for t in t2]
    lines!(ax, getindex.(edge2, 1), getindex.(edge2, 2), getindex.(edge2, 3);
           color = :black, linewidth = 2)

    # Edge 3: B → C
    Cn = [1.0, 1.0, 1.0] / √3
    edge3 = [normalize((1 - t) * Bn + t * Cn) for t in t2]
    lines!(ax, getindex.(edge3, 1), getindex.(edge3, 2), getindex.(edge3, 3);
           color = :black, linewidth = 2)

    # Edge 4: A → C
    edge4 = [normalize((1 - t) * A + t * Cn) for t in t2]
    lines!(ax, getindex.(edge4, 1), getindex.(edge4, 2), getindex.(edge4, 3);
           color = :black, linewidth = 2)

    # seed points
    scatter!(ax,
        getindex.(cart_pts, 1), getindex.(cart_pts, 2), getindex.(cart_pts, 3);
        color = :red, markersize = 8,
    )

    return fig
end

function plot_full_sphere(full_pts)
    fig = Figure(size = (900, 900))
    ax  = Axis3(fig[1, 1];
        aspect = :data,
        xlabel = "x", ylabel = "y", zlabel = "z",
        title  = "Oh full-sphere points ($(length(full_pts)) pts)",
    )

    # translucent sphere
    θs = range(0, π;  length = 60)
    ϕs = range(0, 2π; length = 120)
    xs = [sin(θ) * cos(ϕ) for θ in θs, ϕ in ϕs]
    ys = [sin(θ) * sin(ϕ) for θ in θs, ϕ in ϕs]
    zs = [cos(θ)          for θ in θs, ϕ in ϕs]
    surface!(ax, xs, ys, zs; color = fill((:lightblue, 0.1), size(xs)),
             shading = NoShading, transparency = true)

    scatter!(ax,
        getindex.(full_pts, 1), getindex.(full_pts, 2), getindex.(full_pts, 3);
        color = :red, markersize = 5,
    )

    return fig
end

# Show plots
fig1 = plot_oh_wedge_on_sphere(q)
display(fig1)

fig2 = plot_full_sphere(full_sphere_pts)
display(fig2)
