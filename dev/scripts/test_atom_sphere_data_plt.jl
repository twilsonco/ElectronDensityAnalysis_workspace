#!/usr/bin/env julia

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter
using StaticArrays
using LinearAlgebra
using Delaunay: delaunay

println("Testing AtomSphereData PLT output...")

# Create mock AtomSphereData for testing
function create_mock_atom_sphere_data()
    # Create sphere points using the library function
    center = [0.0, 0.0, 0.0]
    radius = 1.0
    num_points = 20  # Small number for testing

    # Generate points on sphere
    sphere_points_matrix = points_on_sphere_regular(num_points, center, radius)

    # Convert to Vector{SVector{3,Float64}} format
    sphere_points = [SVector{3,Float64}(sphere_points_matrix[i, :]) for i in 1:num_points]

    # Create spherical coordinates (θ, φ) for each point
    sphere_coordinates = [
        SVector{2,Float64}(
            atan(sqrt(pt[1]^2 + pt[2]^2), pt[3]),  # θ (polar angle)
            atan(pt[2], pt[1]),                      # φ (azimuthal angle)
        ) for pt in sphere_points
    ]

    sphere_point_areas = ones(Float64, num_points) * (4π / num_points)  # Equal area distribution

    # Create Delaunay triangulation
    triangulation = delaunay(sphere_points_matrix)

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

# Test 1: AtomSphereData with node-centered data
println("\n=== Test 1: Node-centered data ===")
asd = create_mock_atom_sphere_data()
num_nodes = length(asd.sphere_points)
num_functions = 3

# Create sample function data (F × N matrix)
node_data = Matrix{Float64}(undef, num_functions, num_nodes)
for i in 1:num_functions, j in 1:num_nodes
    # Simple test functions
    if i == 1
        node_data[i, j] = norm(asd.sphere_points[j])  # Distance from origin
    elseif i == 2
        node_data[i, j] = asd.sphere_points[j][1]    # X coordinate
    else
        node_data[i, j] = sin(asd.sphere_coordinates[j][1])  # Function of theta
    end
end

try
    filename = "test_atom_sphere_node_centered.plt"
    zones = [("AtomSphere_NodeCentered", asd, node_data)]

    TecplotWriter.open_plt_file(
        filename,
        "AtomSphereData Node-Centered Test",
        "X, Y, Z, Function1, Function2, Function3",
    )

    # Call the main method with data matrix
    TecplotWriter.write_zone(
        "AtomSphere_NodeCentered",
        asd,
        node_data;
        variables=["X", "Y", "Z", "Function1", "Function2", "Function3"],
        aux_data=["TestType=NodeCentered", "NumFunctions=$num_functions"],
        cell_centered_data=false,
    )

    TecplotWriter.close_plt_file()
    println("✅ Node-centered AtomSphereData test completed successfully!")
    println("   Created file: $filename")

catch e
    println("❌ Node-centered test failed:")
    showerror(stdout, e)
    println()
end

# Test 2: AtomSphereData with cell-centered data (expected to fail for FE zones)
println("\n=== Test 2: Cell-centered data (Note: FE zones require node-centered data) ===")
convex_hull = asd.sphere_triangulation.convex_hull
num_elements = size(convex_hull, 1)

# Create cell-centered data (F × E matrix)
cell_data = Matrix{Float64}(undef, num_functions, num_elements)
for i in 1:num_functions, j in 1:num_elements
    # Compute element center and assign function values
    element_nodes = convex_hull[j, :]  # Get the 3 node indices for this element
    center =
        (
            asd.sphere_points[element_nodes[1]] +
            asd.sphere_points[element_nodes[2]] +
            asd.sphere_points[element_nodes[3]]
        ) / 3

    if i == 1
        cell_data[i, j] = norm(center)
    elseif i == 2
        cell_data[i, j] = center[1]
    else
        cell_data[i, j] = center[2] + center[3]
    end
end

try
    filename = "test_atom_sphere_cell_centered.plt"

    TecplotWriter.open_plt_file(
        filename,
        "AtomSphereData Cell-Centered Test",
        "X, Y, Z, Function1, Function2, Function3",
    )

    try
        # Call the main method with cell-centered data (this should fail for FE zones)
        TecplotWriter.write_zone(
            "AtomSphere_CellCentered",
            asd,
            cell_data;
            variables=["X", "Y", "Z", "Function1", "Function2", "Function3"],
            aux_data=["TestType=CellCentered", "NumFunctions=$num_functions"],
            cell_centered_data=true,
        )

        TecplotWriter.close_plt_file()
        println("⚠️  Cell-centered AtomSphereData test unexpectedly succeeded!")
        println("   Created file: $filename")
    catch inner_e
        # Always close the file, even if write_zone fails
        TecplotWriter.close_plt_file()
        rethrow(inner_e)
    end

catch e
    println(
        "✅ Cell-centered test failed as expected (FE zones require node-centered data):"
    )
    println("   Error: $(typeof(e)): $(e.msg)")
    println(
        "   Note: Use convert_atom_gbd_to_cell_centered() to convert to node-centered data first",
    )
end

# Test 3: Fallback method (geometry only)
println("\n=== Test 3: Geometry-only fallback ===")
geometry_variables = ["X", "Y", "Z"]  # Only coordinate variables
println("Geometry variables: $(geometry_variables)")

try
    filename = "test_atom_sphere_geometry_only.plt"

    TecplotWriter.open_plt_file(
        filename, "AtomSphereData Geometry Only Test", join(geometry_variables, ", ")
    )

    try
        # Call the fallback method without data matrix
        TecplotWriter.write_zone(
            "AtomSphere_GeometryOnly",
            asd;
            variables=geometry_variables,  # Use the restricted variable list
            aux_data=["TestType=GeometryOnly"],
        )

        TecplotWriter.close_plt_file()
        println("✅ Geometry-only AtomSphereData test completed successfully!")
        println("   Created file: $filename")
    catch inner_e
        # Always close the file, even if write_zone fails
        TecplotWriter.close_plt_file()
        rethrow(inner_e)
    end

catch e
    println("❌ Geometry-only test failed:")
    showerror(stdout, e)
    println()
end

println("\n🎉 All AtomSphereData PLT tests completed!")
println("Files created in current directory:")
println("  - test_atom_sphere_node_centered.plt")
println("  - test_atom_sphere_cell_centered.plt")
println("  - test_atom_sphere_geometry_only.plt")
