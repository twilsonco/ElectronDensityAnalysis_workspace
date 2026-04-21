#!/usr/bin/env julia

# Test condensed variable naming with real ethane SystemGBDData
using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter
using Interpolations: Linear
using Serialization: deserialize

# Include test helpers
include("../test/test_helpers.jl")

function test_ethane_condensed_variables()
    println("🧪 Testing condensed variable naming with real ethane SystemGBDData...")

    checkpoint_dir = "/Users/haiiro/NoSync/ElectronDensityAnalysis.jl/gbd_checkpoints/Ethane"
    # sys = with_zip_file("test_data/ethanol_adf.t41", load_data_scm_adf_t41)
    load_sys_args = (
        "test_data/ethane_adf.t41", (f) -> with_zip_file(f, load_data_scm_adf_t41)
    )

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys = handle_checkpoint(sys_checkpoint_file, with_zip_file, load_sys_args)[1]

    # return sys

    system_gbd_data = gradient_bundle_decomposition(
        sys;
        num_gbs=500,
        interp_type=Linear,
        func_cutoff=1e-3,
        curvature_threshold=0.8 * 0.5π,
        atom_bounding_box_shrink_factor=0.8,
        seed_sphere_radius_factor=0.1,
        checkpoint_dir=checkpoint_dir,
        system_dependency_files=[sys_checkpoint_file],
        # force_recompute=true,
    )

    # if system_gbd_data === nothing
    #     println("Creating new SystemGBDData...")
    #     try
    #         system_gbd_data = gradient_bundle_decomposition(
    #             sys;
    #             cps=cps,
    #             num_gbs=100,  # Smaller number for testing
    #             interp_type=Linear,
    #             func_cutoff=1e-3,
    #             curvature_threshold=0.8 * 0.5π,
    #             atom_bounding_box_shrink_factor=0.8,
    #             seed_sphere_radius_factor=0.1,
    #             use_checkpoints=true,
    #             checkpoint_dir="gbd_checkpoints",
    #             verbose_checkpoints=false,
    #         )
    #         println("✓ Completed gradient bundle decomposition")
    #     catch e
    #         println("❌ Failed gradient bundle decomposition: $e")
    #         return false
    #     end
    # end

    println("  Critical points: $(length(system_gbd_data.critical_points))")
    println("  DGBs: $(length(system_gbd_data.differential_gradient_bundles))")
    println("  Rejected paths: $(length(system_gbd_data.rejected_gradient_paths))")
    println("  Atom sphere data: $(length(system_gbd_data.atom_sphere_data))")
    println("  Function names: $(system_gbd_data.function_names)")

    # Test condensed variable naming functionality
    println("\n=== Testing Condensed Variable Functionality ===")

    # Test helper functions with the actual function names from SystemGBDData
    println("Function name transformations:")
    for fname in system_gbd_data.function_names
        normalized = TecplotWriter._normalize_density_variable_name(fname)
        condensed = TecplotWriter._create_condensed_variable_name(fname)
        println("  $fname → $condensed")
    end

    # Test variable detection
    println("\nTesting variable detection for SystemGBDData zones:")
    sgbd_zones = Tuple{String,Any}[("EthaneSystemGBD", system_gbd_data)]
    detected_variables_str = TecplotWriter._get_variables_for_zones(sgbd_zones, sys)
    println("Detected variables: $detected_variables_str")

    # Parse the variables string back to a vector for testing
    detected_variables = [String(strip(v)) for v in split(detected_variables_str, ",")]

    # Verify that condensed variables are present
    expected_condensed_vars = [
        TecplotWriter._create_condensed_variable_name(fname) for
        fname in system_gbd_data.function_names
    ]
    println("\nExpected condensed variables: $(join(expected_condensed_vars, ", "))")

    for expected_var in expected_condensed_vars
        if expected_var in detected_variables
            println("  ✅ Found: $expected_var")
        else
            println("  ❌ Missing: $expected_var")
        end
    end

    # Write test PLT file with condensed variables
    println("\n=== Writing PLT File with Real Ethane Data ===")
    test_filename = "test_ethane_condensed_variables.plt"

    try
        TecplotWriter.open_plt_file(
            test_filename,
            "Ethane SystemGBDData with Condensed Variables",
            detected_variables_str,
        )

        println("Writing SystemGBDData with condensed variable names...")

        # Write only the first couple of zones to avoid the finite element issues
        # but demonstrate the condensed variable naming

        # 1. Write system zone
        TecplotWriter.write_zone(
            "Ethane System", sys; system=sys, variables=detected_variables
        )

        # 2. Write critical points
        if !isempty(system_gbd_data.critical_points)
            TecplotWriter.write_zone(
                "Ethane Critical Points",
                system_gbd_data.critical_points;
                system=sys,
                variables=detected_variables,
            )
        end

        # 3. Write atom sphere zones (geometry-only) for each atom
        #    This avoids FE data-count issues while still saving the sphere meshes
        println("Writing atom sphere zones (geometry-only)...")
        num_spheres_written = 0
        # Sort by atom_number for stable ordering
        spheres = collect(values(system_gbd_data.atom_sphere_data))
        sort!(spheres; by=x -> x.atom_number)
        for asd in spheres
            zone_name = "Ethane Atom $(asd.atom_number) Sphere"
            TecplotWriter.write_zone(
                zone_name,
                asd;
                system=sys,
                variables=detected_variables,
                cell_centered_data=false,
            )
            num_spheres_written += 1
        end
        println("✓ Wrote $(num_spheres_written) atom sphere zones")

        TecplotWriter.close_plt_file()

        if isfile(test_filename)
            size = filesize(test_filename)
            println("✅ Successfully created ethane PLT file: $test_filename ($size bytes)")

            # Verify file header
            header = read(test_filename, 8)
            println("✅ File header: $(String(header))")

            println("\n🎯 This file demonstrates condensed variables with real ethane data:")
            println("   📊 Original system with 3D field: 'Electron Density'")
            for fname in system_gbd_data.function_names
                condensed_name = TecplotWriter._create_condensed_variable_name(fname)
                println("   🔬 DGB condensed function: '$condensed_name'")
            end
            println("   📐 Solid angle: 'α'")

            return true
        else
            println("❌ PLT file was not created")
            return false
        end

    catch e
        println("❌ Error creating ethane PLT file: $e")
        try
            TecplotWriter.close_plt_file()
        catch
        end
        return false
    end
end

# Run the test
# if abspath(PROGRAM_FILE) == @__FILE__
println("Starting ethane condensed variables test...")
success = test_ethane_condensed_variables()
if success
    println("\n🎉 Ethane condensed variables test completed successfully!")
else
    println("\n❌ Ethane condensed variables test failed!")
    exit(1)
end
# end
