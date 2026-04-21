#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using Interpolations: Linear

# Create minimal test using existing data loaders and test utilities
include("../test/test_helpers.jl")

# Test SystemGBDData writing with full GBD analysis (like test_gbd_per_atom.jl)
function test_full_system_gbd()
    println("\n=== Full SystemGBDData Test with GBD Analysis ===")

    # Load ethane system (small but realistic)
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

    # Find critical points
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

    # Perform gradient bundle decomposition
    println("\nPerforming gradient bundle decomposition...")
    system_gbd_data = nothing
    try
        system_gbd_data = gradient_bundle_decomposition(
            sys;
            cps=cps,
            num_gbs=100,                     # Very small number for testing
            interp_type=Linear,
            func_cutoff=1e-3,
            curvature_threshold=0.8 * 0.5π,
            atom_bounding_box_shrink_factor=0.8,
            seed_sphere_radius_factor=0.1,
        )
        println("✓ Completed gradient bundle decomposition")
        println("  Critical points: $(length(system_gbd_data.critical_points))")
        println("  DGBs: $(length(system_gbd_data.differential_gradient_bundles))")
        println("  Rejected paths: $(length(system_gbd_data.rejected_gradient_paths))")
        println("  Atom sphere data: $(length(system_gbd_data.atom_sphere_data))")
        println("  Function names: $(system_gbd_data.function_names)")
    catch e
        println("✗ Failed gradient bundle decomposition: $e")
        showerror(stdout, e)
        println()
        return false
    end

    # Test 1: Write SystemGBDData with geometry-only spheres
    println("\n--- Test 1: Geometry-only SystemGBDData ---")
    variables = ["X", "Y", "Z", "ρ", "V", "α"]  # Include function names and solid angle

    try
        open_plt_file(
            "test_full_system_gbd_geometry.plt",
            "Full SystemGBDData Geometry",
            join(variables, ", "),
        )

        write_zone(
            "Full SystemGBD",
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
        close_plt_file()  # Ensure file is closed
        return false
    end

    # Test 2: Write SystemGBDData with cell-centered spheres (if conversion works)
    println("\n--- Test 2: Cell-centered SystemGBDData ---")
    variables_with_functions = ["X", "Y", "Z", "ρ", "V", "α"]  # Use same variables

    try
        open_plt_file(
            "test_full_system_gbd_cell_centered.plt",
            "Full SystemGBDData Cell-Centered",
            join(variables_with_functions, ", "),
        )

        write_zone(
            "Full SystemGBD CC",
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
        close_plt_file()  # Ensure file is closed
        return false
    end

    return true
end

# Run the comprehensive test
println("Starting comprehensive SystemGBDData test with full GBD analysis...")

test_passed = test_full_system_gbd()

println("\n=== Test Result ===")
if test_passed
    println("Full SystemGBDData test PASSED! ✓")
    println("The SystemGBDData write_zone method successfully handles:")
    println("  ✓ System grid data")
    println("  ✓ Critical points")
    println("  ✓ Atom sphere data with triangulated surfaces")
    println("  ✓ Differential gradient bundles")
    println("  ✓ Rejected gradient paths")
    println("  ✓ Both geometry-only and cell-centered sphere data")
else
    println("Full SystemGBDData test FAILED! ✗")
end
