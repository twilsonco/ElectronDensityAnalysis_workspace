#!/usr/bin/env julia
"""
Demo script for SymmetrySpherePoints — generates and visualizes seed points
for Oh and Td symmetry groups.
"""

include("SymmetrySpherePoints.jl")
using .SymmetrySpherePoints
using LinearAlgebra
using Printf
using GLMakie

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: generic 3D plot for any symmetry group
# ═══════════════════════════════════════════════════════════════════════════════

function plot_wedge_on_sphere(g::SymmetryGroup, q::Int; interior_only::Bool = false)
    pts      = wedge_points(g, q; interior_only)
    cart_pts = [p[1] for p in pts]
    A, B, C  = wedge_vertices(g)

    label = interior_only ? "interior" : "all"
    gname = nameof(typeof(g))

    fig = Figure(size = (900, 900))
    ax  = Axis3(fig[1, 1];
        aspect = :data,
        xlabel = "x", ylabel = "y", zlabel = "z",
        title  = "$gname wedge  (q=$q, $label, $(length(cart_pts)) pts)",
    )

    # translucent sphere
    θs = range(0, π;  length = 60)
    ϕs = range(0, 2π; length = 120)
    xs = [sin(θ) * cos(ϕ) for θ in θs, ϕ in ϕs]
    ys = [sin(θ) * sin(ϕ) for θ in θs, ϕ in ϕs]
    zs = [cos(θ)          for θ in θs, ϕ in ϕs]
    surface!(ax, xs, ys, zs; color = fill((:lightblue, 0.12), size(xs)),
             shading = NoShading, transparency = true)

    # wedge boundary edges (geodesic arcs via linear interpolation + normalize)
    t = range(0, 1; length = 100)
    for (P, Q) in ((A, B), (B, C), (A, C))
        edge = [normalize((1 - s) * P + s * Q) for s in t]
        lines!(ax, getindex.(edge, 1), getindex.(edge, 2), getindex.(edge, 3);
               color = :black, linewidth = 2)
    end

    # seed points
    scatter!(ax,
        getindex.(cart_pts, 1), getindex.(cart_pts, 2), getindex.(cart_pts, 3);
        color = :red, markersize = 8,
    )

    return fig
end

function plot_full_sphere(g::SymmetryGroup, full_pts)
    gname = nameof(typeof(g))

    fig = Figure(size = (900, 900))
    ax  = Axis3(fig[1, 1];
        aspect = :data,
        xlabel = "x", ylabel = "y", zlabel = "z",
        title  = "$gname full sphere  ($(length(full_pts)) pts)",
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

# ═══════════════════════════════════════════════════════════════════════════════
# Run for both groups
# ═══════════════════════════════════════════════════════════════════════════════

q = 16

for g in (OhSymmetry(), TdSymmetry())
    gname = nameof(typeof(g))
    println("\n── $gname  (order $(group_order(g))) ──")

    wpts = wedge_points(g, q)
    fpts = full_sphere_points(g, q)
    println("  Wedge points (all):      $(length(wpts))")
    println("  Full-sphere points:      $(length(fpts))")

    wpts_int = wedge_points(g, q; interior_only = true)
    fpts_int = full_sphere_points(g, q; interior_only = true)
    println("  Wedge points (interior): $(length(wpts_int))")
    println("  Full-sphere (interior):  $(length(fpts_int))")

    display(plot_wedge_on_sphere(g, q))
    display(plot_wedge_on_sphere(g, q; interior_only = true))
    display(plot_full_sphere(g, fpts))
end
