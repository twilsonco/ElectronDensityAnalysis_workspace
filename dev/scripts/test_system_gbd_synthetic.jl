#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using ElectronDensityAnalysis:
    SystemGBDData, AtomSphereData, DifferentialGradientBundle, GradientPath, CriticalPoint
using Delaunay
using LinearAlgebra

# Create test using existing data loaders but synthetic analysis data
include("../test/test_helpers.jl")

# Test SystemGBDData writing with real system data + synthetic analysis data
function test_synthetic_system_gbd()
    println("\n=== SystemGBDData Test with Real System + Synthetic Analysis ===")

    # Load real system (ethane)
    println("Loading ethane system...")
    sys = nothing
    try
        sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
        println("✓ Successfully loaded ethane system")
        println("  System: $(sys.name)")
        println("  Grid: $(sys.grid.n_pts)")
        println("  Atoms: $(length(sys.atoms))")
    catch e
        println("✗ Failed to load ethane data: $e")
        return false
    end

    # Find real critical points
    println("\nFinding critical points...")
    cps = nothing
    try
        cps = find_critical_points(sys; spacing=0.35, dup_cp_check_dist=0.1)
        println("✓ Found $(length(cps)) critical points")

        # Show CP types
        cp_counts = Dict()
        for cp in cps
            cp_counts[cp.type] = get(cp_counts, cp.type, 0) + 1
        end
        for (cp_type, count) in cp_counts
            println("  $cp_type: $count")
        end
    catch e
        println("✗ Failed to find critical points: $e")
        return false
    end

    # Create synthetic analysis data for SystemGBDData
    println("\nCreating synthetic analysis data...")

    # Find nuclear CPs (atoms)
    nuclear_cps = [cp for cp in cps if cp.type == nuclear_cp]
    println("Found $(length(nuclear_cps)) nuclear critical points")

    # Create synthetic AtomSphereData for first few atoms
    atom_sphere_data = Dict{Int,AtomSphereData}()
    dgbs = DifferentialGradientBundle[]
    rejected_paths = GradientPath[]

    for (i, ncp) in enumerate(nuclear_cps[1:min(3, length(nuclear_cps))])  # Just first 3 atoms
        cp_index = i
        atom_num = i

        # Create synthetic sphere points around the nuclear CP
        center = ncp.r
        radius = 0.5

        # Create more points for proper triangulation (icosphere-like)
        sphere_points = []

        # Add regular icosahedron vertices scaled to radius
        phi = (1.0 + sqrt(5.0)) / 2.0  # golden ratio

        vertices = [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]

        for v in vertices
            # Normalize to unit sphere then scale by radius
            norm_v = v / norm(v)
            push!(sphere_points, center + radius * norm_v)
        end

        println(
            "  Atom $atom_num: Created $(length(sphere_points)) sphere points around $(center)",
        )

        # Create triangulation
        points_matrix = hcat([Float64.(pt) for pt in sphere_points]...)  # 3×8 matrix
        tri = delaunay(points_matrix)

        # Create synthetic gradient paths
        gradient_paths = []
        for j in 1:3  # 3 synthetic paths per atom
            path_points = hcat(
                center, center + 0.1 * [j-2, 0.1, 0.1], center + 0.2 * [j-2, 0.2, 0.2]
            )
            gp = GradientPath(path_points, cp_index, 0, "converged")
            push!(gradient_paths, gp)

            # Create DGB from this path
            dgb = DifferentialGradientBundle(
                cp_index,
                0.1 * j,  # theta 
                0.2 * j,  # phi
                gp,
                [1.0 + 0.1*j, 2.0 + 0.1*j],  # function values
                [0.5 + 0.05*j],  # solid angle
            )
            push!(dgbs, dgb)
        end

        # Create AtomSphereData
        asd = AtomSphereData(atom_num, cp_index, radius, sphere_points, tri, gradient_paths)
        atom_sphere_data[cp_index] = asd

        # Create some rejected paths
        if i == 1  # Only for first atom
            rejected_path = GradientPath(
                hcat(center, center + [-0.1, -0.1, -0.1]), cp_index, -1, "max_steps"
            )
            push!(rejected_paths, rejected_path)
        end
    end

    # Create SystemGBDData with synthetic data
    function_names = ["P", "T"]  # Charge density and kinetic energy density
    function_totals = [10.5, 25.3]  # Synthetic totals

    system_gbd_data = SystemGBDData(
        sys, cps, function_names, function_totals, dgbs, atom_sphere_data, rejected_paths
    )

    println("✓ Created SystemGBDData with:")
    println("  Critical points: $(length(system_gbd_data.critical_points))")
    println("  DGBs: $(length(system_gbd_data.differential_gradient_bundles))")
    println("  Rejected paths: $(length(system_gbd_data.rejected_gradient_paths))")
    println("  Atom sphere data: $(length(system_gbd_data.atom_sphere_data))")
    println("  Function names: $(system_gbd_data.function_names)")

    # Test 1: Write SystemGBDData with geometry-only spheres
    println("\n--- Test 1: Geometry-only SystemGBDData ---")
    variables = ["X", "Y", "Z", "Electron Density"]

    try
        open_plt_file(
            "test_synthetic_system_gbd_geometry.plt",
            "Synthetic SystemGBDData Geometry",
            join(variables, ", "),
        )

        write_zone(
            "Synthetic SystemGBD",
            system_gbd_data;
            system=sys,
            variables=variables,
            cell_centered_spheres=false,
        )

        close_plt_file()
        println("✓ Successfully wrote geometry-only SystemGBDData file")
    catch e
        println("✗ Failed to write geometry-only SystemGBDData file:")
        showerror(stdout, e)
        println()
        close_plt_file()
        return false
    end

    # Test 2: Write SystemGBDData with cell-centered spheres 
    println("\n--- Test 2: Cell-centered SystemGBDData ---")
    variables_with_functions = ["X", "Y", "Z", "Electron Density"]
    append!(variables_with_functions, system_gbd_data.function_names)

    try
        open_plt_file(
            "test_synthetic_system_gbd_cell_centered.plt",
            "Synthetic SystemGBDData Cell-Centered",
            join(variables_with_functions, ", "),
        )

        write_zone(
            "Synthetic SystemGBD CC",
            system_gbd_data;
            system=sys,
            variables=variables_with_functions,
            cell_centered_spheres=true,
        )

        close_plt_file()
        println("✓ Successfully wrote cell-centered SystemGBDData file")
    catch e
        println("✗ Failed to write cell-centered SystemGBDData file:")
        showerror(stdout, e)
        println()
        close_plt_file()
        return false
    end

    return true
end

# Run the test
println("Starting SystemGBDData test with real system data and synthetic analysis...")

test_passed = test_synthetic_system_gbd()

println("\n=== Test Result ===")
if test_passed
    println("Synthetic SystemGBDData test PASSED! ✓")
    println("The SystemGBDData write_zone method successfully handles:")
    println("  ✓ Real system grid data (ethane)")
    println("  ✓ Real critical points")
    println("  ✓ Synthetic atom sphere data with triangulated surfaces")
    println("  ✓ Synthetic differential gradient bundles")
    println("  ✓ Synthetic rejected gradient paths")
    println("  ✓ Both geometry-only and cell-centered sphere data")
    println("\nThis demonstrates the full SystemGBDData multi-zone output structure!")
else
    println("Synthetic SystemGBDData test FAILED! ✗")
end
