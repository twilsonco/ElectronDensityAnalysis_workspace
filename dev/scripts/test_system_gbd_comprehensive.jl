#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
include("../test/test_helpers.jl")

# Create comprehensive test using existing ethane system with minimal gradient bundles
function test_system_gbd_comprehensive()
    println("Starting comprehensive SystemGBD test using ethane system...")

    # Use existing working ethane system with minimal gradient bundles for faster computation
    try
        sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
        println("✓ Successfully loaded ethane system")
        println("  System: $(sys.name)")
        println("  Atoms: $(length(sys.atoms))")

        # Find critical points
        cps = find_critical_points(sys; spacing=0.35, dup_cp_check_dist=0.1)
        println("✓ Found $(length(cps)) critical points")

        # Print CP types breakdown
        cp_types = Dict{Any,Int}()
        for cp in cps
            cp_types[cp.type] = get(cp_types, cp.type, 0) + 1
        end
        for (cp_type, count) in cp_types
            println("  $(string(cp_type)): $count")
        end

        # Run GBD with minimal gradient bundles (20 per atom)
        println("Running gradient bundle decomposition with 20 bundles per atom...")

        system_gbd_data = gradient_bundle_decomposition(
            sys;
            cps=cps,
            num_gbs=20,  # Small number for fast completion
            gp_num_steps=200,  # Reduce steps for speed
            func_cutoff=1e-2,  # Less strict cutoff
            seed_sphere_radius_factor=0.1,
            curvature_threshold=0.5,
        )

        println("✓ Successfully completed gradient bundle decomposition")
        println("  - $(length(system_gbd_data.atom_sphere_data)) atom spheres")
        println("  - $(length(system_gbd_data.differential_gradient_bundles)) total DGBs")
        println("  - $(length(system_gbd_data.rejected_gradient_paths)) rejected paths")

        # Write comprehensive SystemGBD zone files 
        write_system_gbd_files(system_gbd_data)

        println("✓ Test completed successfully!")

    catch e
        println("❌ Error during comprehensive test: $e")
        throw(e)
    end
end

function write_system_gbd_files(system_gbd_data)
    println("Writing SystemGBD zone files...")

    # Prepare variables list including function names
    variables = ["X", "Y", "Z"]
    for fname in system_gbd_data.function_names
        push!(variables, fname)
    end

    # 1. Basic geometry and functions test
    try
        open_plt_file(
            "test_comprehensive_system_gbd_complete.plt",
            "Comprehensive SystemGBDData Complete",
            join(variables, ", "),
        )

        write_zone("Comprehensive SystemGBD", system_gbd_data)

        close_plt_file()
        println("✓ Written complete file: test_comprehensive_system_gbd_complete.plt")

    catch e
        println("❌ Error writing complete file: $e")
        close_plt_file()
        rethrow(e)
    end

    # 2. With cell-centered spheres test  
    try
        open_plt_file(
            "test_comprehensive_system_gbd_cell_centered.plt",
            "Comprehensive SystemGBDData Cell-Centered",
            join(variables, ", "),
        )

        write_zone(
            "Comprehensive SystemGBD CC", system_gbd_data; cell_centered_spheres=true
        )

        close_plt_file()
        println(
            "✓ Written cell-centered file: test_comprehensive_system_gbd_cell_centered.plt"
        )

    catch e
        println("❌ Error writing cell-centered file: $e")
        close_plt_file()
        rethrow(e)
    end
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_system_gbd_comprehensive()
end
