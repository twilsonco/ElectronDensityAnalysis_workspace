#!/usr/bin/env julia

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter
using StaticArrays
using LinearAlgebra
using Delaunay: delaunay

println("Debugging AtomSphereData PLT output...")

# Create simple AtomSphereData for debugging
function create_simple_atom_sphere_data()
    # Create sphere points using the library function
    center = [0.0, 0.0, 0.0]
    radius = 1.0
    num_points_requested = 8  # Request 8 points

    # Generate points on sphere (might return more than requested)
    sphere_points_matrix = points_on_sphere_regular(num_points_requested, center, radius)
    actual_num_points = size(sphere_points_matrix, 1)  # Use actual number returned

    # Convert to Vector{SVector{3,Float64}} format
    sphere_points = [
        SVector{3,Float64}(sphere_points_matrix[i, :]) for i in 1:actual_num_points
    ]

    # Create spherical coordinates (θ, φ) for each point
    sphere_coordinates = [
        SVector{2,Float64}(
            atan(sqrt(pt[1]^2 + pt[2]^2), pt[3]),  # θ (polar angle)
            atan(pt[2], pt[1]),                      # φ (azimuthal angle)
        ) for pt in sphere_points
    ]

    sphere_point_areas = ones(Float64, actual_num_points) * (4π / actual_num_points)  # Equal area distribution

    # Create Delaunay triangulation
    triangulation = delaunay(sphere_points_matrix)

    println("Created AtomSphereData with:")
    println("  - $(actual_num_points) nodes (requested $(num_points_requested))")
    println("  - $(size(triangulation.convex_hull, 1)) elements")
    println(
        "  - Node indices range: $(minimum(triangulation.convex_hull)) to $(maximum(triangulation.convex_hull))",
    )

    return AtomSphereData(
        1,                    # atom_number
        1,                    # critical_point_number
        radius,               # sphere_radius
        triangulation,        # sphere_triangulation (proper Triangulation object)
        sphere_point_areas,   # sphere_point_areas
        sphere_coordinates,   # sphere_coordinates
        sphere_points,         # sphere_points
    )
end

# Test: Simple coordinates only
println("\n=== Debug Test: Coordinates Only ===")
asd = create_simple_atom_sphere_data()
num_nodes = length(asd.sphere_points)

# Just X, Y, Z coordinates - no function data
variables = ["X", "Y", "Z"]
println("Variables: $(variables)")
println(
    "Expected data points: $(num_nodes) nodes × $(length(variables)) variables = $(num_nodes * length(variables))",
)

try
    filename = "debug_coordinates_only.plt"

    TecplotWriter.open_plt_file(filename, "Debug Coordinates Only", join(variables, ", "))

    # Create empty data matrix (3 functions × num_nodes, but we only use coordinates)
    empty_data = zeros(Float64, 0, num_nodes)  # 0 functions

    # Call the main method with empty data matrix
    TecplotWriter.write_zone(
        "CoordinatesOnly",
        asd,
        empty_data;
        variables=variables,
        aux_data=["TestType=CoordinatesOnly"],
        cell_centered_data=false,
    )

    TecplotWriter.close_plt_file()
    println("✅ Coordinates-only test completed successfully!")
    println("   Created file: $filename")

catch e
    println("❌ Coordinates-only test failed:")
    showerror(stdout, e)
    println()
end

# Test: One function
println("\n=== Debug Test: One Function ===")
variables = ["X", "Y", "Z", "TestFunction"]
println("Variables: $(variables)")

# Create 1 function × num_nodes data matrix
one_func_data = Matrix{Float64}(undef, 1, num_nodes)
for j in 1:num_nodes
    one_func_data[1, j] = norm(asd.sphere_points[j])  # Distance from origin
end

try
    filename = "debug_one_function.plt"

    TecplotWriter.open_plt_file(filename, "Debug One Function", join(variables, ", "))

    # Call the main method with 1 function
    TecplotWriter.write_zone(
        "OneFunction",
        asd,
        one_func_data;
        variables=variables,
        aux_data=["TestType=OneFunction"],
        cell_centered_data=false,
    )

    TecplotWriter.close_plt_file()
    println("✅ One function test completed successfully!")
    println("   Created file: $filename")

catch e
    println("❌ One function test failed:")
    showerror(stdout, e)
    println()
end

println("\n🎉 Debug tests completed!")
