#!/usr/bin/env julia

using ElectronDensityAnalysis
using StaticArrays
using Delaunay
using Interpolations
using LinearAlgebra
include("test/test_helpers.jl")

println("🧪 Testing SystemGBDData write_zone functionality...")

# Helper function to create a simple synthetic system for testing
function create_test_system(name="Test System")
    origin = [0.0, 0.0, 0.0]
    lattice = [0.1 0.0 0.0; 0.0 0.1 0.0; 0.0 0.0 0.1]
    n_pts = [5, 5, 5]  # Small grid for fast testing
    gs = GridSpec(origin, lattice, n_pts)

    atoms = [
        NuclearCoordinate([0.2, 0.2, 0.2], 6),   # Carbon
        NuclearCoordinate([0.3, 0.2, 0.2], 1),   # Hydrogen
    ]

    # Create simple test data
    data = zeros(Float32, n_pts...)
    for i in 1:n_pts[1], j in 1:n_pts[2], k in 1:n_pts[3]
        pos = [i-1, j-1, k-1] .* 0.1
        for atom in atoms
            r = norm(pos - atom.r)
            amplitude = atom.data.number == 6 ? 2.0 : 0.5
            data[i, j, k] += amplitude * exp(-r^2 / 0.2)
        end
    end

    # Create interpolation functions
    g = generate_interpolation_grid(gs)
    itp = interpolate((g[1], g[2], g[3]), data, Gridded(Linear()))

    func_xyz = (r) -> itp(r...)
    func_single = (r) -> func_xyz(r)
    grad_func = (r) -> [0.0, 0.0, 0.0]  # Dummy
    hess_func = (r) -> diagm([1.0, 1.0, 1.0])  # Dummy

    func! = (result, r) -> result[1] = func_single(r)
    grad! = (result, r) -> result .= grad_func(r)
    hess! = (result, r) -> result .= hess_func(r)

    return System(
        name,
        "Test System for SystemGBDData",
        atoms,
        gs,
        data,
        [false, false, false],
        func_xyz,
        func_single,
        grad_func,
        hess_func,
        func!,
        grad!,
        hess!,
    )
end

# Create synthetic SystemGBDData for testing
function create_test_system_gbd_data()
    # Create a simple test system
    system = create_test_system()

    # Create test critical points
    critical_points = [
        CriticalPoint([0.2, 0.2, 0.2], CriticalPointType(-3), 1),  # NCP at index 1 (nuclear_cp = -3)
        CriticalPoint([0.25, 0.2, 0.2], CriticalPointType(-1), 0),     # BCP at index 2 (bond_cp = -1)
    ]

    # Create function names
    function_names = ["Density", "Kinetic Energy", "Potential Energy"]
    function_totals = [10.5, 25.3, -35.8]  # System-wide totals

    # Create test AtomSphereData for atom 1
    center = [0.2, 0.2, 0.2]
    radius = 1.0
    num_points = 8  # Small number for testing

    sphere_points_matrix = points_on_sphere_regular(num_points, center, radius)
    sphere_points = [SVector{3,Float64}(sphere_points_matrix[i, :]) for i in 1:num_points]

    sphere_coordinates = [
        SVector{2,Float64}(atan(sqrt(pt[1]^2 + pt[2]^2), pt[3]), atan(pt[2], pt[1])) for
        pt in sphere_points
    ]

    sphere_point_areas = ones(Float64, num_points) * (4π / num_points)

    # Use proper Delaunay triangulation
    triangulation = delaunay(sphere_points_matrix)
    println("Triangulation convex_hull shape: $(size(triangulation.convex_hull))")
    println("Triangulation convex_hull max value: $(maximum(triangulation.convex_hull))")
    println("Number of sphere points: $num_points")

    atom_sphere_data = AtomSphereData(
        1, 1, radius, triangulation, sphere_point_areas, sphere_coordinates, sphere_points
    )

    # Create test DGBs
    dgbs = Vector{DifferentialGradientBundle}()
    for i in 1:3  # Create 3 DGBs for testing
        # Create a simple gradient path
        path_points = zeros(3, 5)  # 5 points along path
        for j in 1:5
            t = (j-1) / 4.0
            path_points[:, j] = center .+ [0.05*t, 0.0, 0.0]
        end

        gp = GradientPath(path_points, 1, 2, TerminationStatus(3))  # From NCP (1) to BCP (2), terminated_at_cp=3

        # Create DGB with some test function values
        dgb = DifferentialGradientBundle(gp, 1, i, 3)  # cp_index=1, sphere_node_index=i, 3 functions

        # Add some test values
        dgb_with_values = DifferentialGradientBundle(
            dgb;
            function_values=[1.2 + i*0.1, 2.5 + i*0.2, -3.1 + i*0.1],
            solid_angle=0.125 + i*0.01,
            area_normalized_values=[0.8 + i*0.05, 1.5 + i*0.1, -2.2 + i*0.05],
        )

        push!(dgbs, dgb_with_values)
    end

    # Create test rejected gradient paths
    rejected_paths = Vector{GradientPath}()
    for i in 1:2  # Create 2 rejected paths for testing
        path_points = zeros(3, 3)  # Shorter paths
        for j in 1:3
            t = (j-1) / 2.0
            path_points[:, j] = center .+ [0.02*t, 0.01*t, 0.0]
        end

        gp = GradientPath(path_points, 1, 0, TerminationStatus(0))  # From NCP (1), rejected (not_terminated=0)
        push!(rejected_paths, gp)
    end

    # Create the atom_sphere_data dictionary
    atom_sphere_dict = Dict{Int,AtomSphereData}(1 => atom_sphere_data)

    # Create SystemGBDData
    sgbd = SystemGBDData(
        system,
        critical_points,
        function_names,
        function_totals,
        dgbs,
        atom_sphere_dict,
        rejected_paths,
    )

    return sgbd
end

println("📊 Creating test SystemGBDData...")
sgbd = create_test_system_gbd_data()

println("✨ SystemGBDData created with:")
println("   - System: $(length(sgbd.system.atoms)) atoms")
println("   - Critical Points: $(length(sgbd.critical_points))")
println(
    "   - Functions: $(length(sgbd.function_names)) ($(join(sgbd.function_names, ", ")))"
)
println("   - DGBs: $(length(sgbd.differential_gradient_bundles))")
println("   - Atom Spheres: $(length(sgbd.atom_sphere_data))")
println("   - Rejected Paths: $(length(sgbd.rejected_gradient_paths))")

# Test 1: Geometry-only spheres (default)
println("\n🔹 Testing SystemGBDData with geometry-only spheres...")
try
    test_filename = "test_system_gbd_geometry.plt"
    variables = ["X", "Y", "Z", "Density", "Kinetic Energy", "Potential Energy"]

    TecplotWriter.open_plt_file(test_filename, "SystemGBDData Test", join(variables, ", "))
    TecplotWriter.write_zone(
        "TestSystem", sgbd; variables=variables, cell_centered_spheres=false
    )
    TecplotWriter.close_plt_file()

    if isfile(test_filename)
        size = filesize(test_filename)
        header = read(test_filename, 8)
        println("   ✅ Geometry-only spheres: $size bytes, header: $(String(header))")
        rm(test_filename)
    else
        println("   ❌ Geometry-only spheres: File not created")
    end
catch e
    println("   ❌ Geometry-only spheres error: $e")
    try
        TecplotWriter.close_plt_file()
    catch
    end
end

# Test 2: Cell-centered spheres  
println("\n🔹 Testing SystemGBDData with cell-centered spheres...")
try
    test_filename = "test_system_gbd_cellcentered.plt"
    variables = ["X", "Y", "Z", "Density", "Kinetic Energy", "Potential Energy"]

    TecplotWriter.open_plt_file(
        test_filename, "SystemGBDData Cell-Centered Test", join(variables, ", ")
    )
    TecplotWriter.write_zone(
        "TestSystem", sgbd; variables=variables, cell_centered_spheres=true
    )
    TecplotWriter.close_plt_file()

    if isfile(test_filename)
        size = filesize(test_filename)
        header = read(test_filename, 8)
        println("   ✅ Cell-centered spheres: $size bytes, header: $(String(header))")
        rm(test_filename)
    else
        println("   ❌ Cell-centered spheres: File not created")
    end
catch e
    println("   ❌ Cell-centered spheres error: $e")
    try
        TecplotWriter.close_plt_file()
    catch
    end
end

println("\n🎉 SystemGBDData testing completed!")
