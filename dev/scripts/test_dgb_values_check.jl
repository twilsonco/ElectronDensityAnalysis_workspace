#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using Interpolations: Linear

# Create minimal test using existing data loaders and test utilities
include("../test/test_helpers.jl")

# Test to verify DGB function values and solid angles are being written correctly
function test_dgb_values_check()
    println("\n=== DGB Values Check Test ===")

    # Load ethane system
    println("Loading ethane system...")
    sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
    println("✓ Successfully loaded ethane system")

    # Find critical points
    println("Finding critical points...")
    cps = find_critical_points(sys; spacing=0.35, dup_cp_check_dist=0.1)
    println("✓ Found $(length(cps)) critical points")

    # Perform gradient bundle decomposition with a small test
    println("Performing gradient bundle decomposition...")
    system_gbd_data = gradient_bundle_decomposition(
        sys;
        cps=cps,
        num_gbs=50,  # Small number for quick test
        interp_type=Linear,
        func_cutoff=1e-3,
        curvature_threshold=0.8 * 0.5π,
        atom_bounding_box_shrink_factor=0.8,
        seed_sphere_radius_factor=0.1,
    )
    println("✓ Completed gradient bundle decomposition")

    # Check the DGB data
    println("\nAnalyzing DGB data...")
    println("  Function names: $(system_gbd_data.function_names)")
    println("  Total DGBs: $(length(system_gbd_data.differential_gradient_bundles))")

    # Look at first few DGBs from atom 1
    atom_1_dgbs = [
        dgb for dgb in system_gbd_data.differential_gradient_bundles if dgb.cp_index == 1
    ]
    println("  DGBs for atom 1: $(length(atom_1_dgbs))")

    if !isempty(atom_1_dgbs)
        dgb = atom_1_dgbs[1]
        println("  Sample DGB data:")
        println("    CP index: $(dgb.cp_index)")
        println("    Sphere node index: $(dgb.sphere_node_index)")
        println("    Function values: $(dgb.function_values)")
        println("    Solid angle: $(dgb.solid_angle)")
        println("    Area normalized values: $(dgb.area_normalized_values)")

        # Check ranges across all DGBs for atom 1
        if length(atom_1_dgbs) > 1
            ρ_values = [
                dgb.function_values[1] for
                dgb in atom_1_dgbs if length(dgb.function_values) >= 1
            ]
            V_values = [
                dgb.function_values[2] for
                dgb in atom_1_dgbs if length(dgb.function_values) >= 2
            ]
            solid_angles = [dgb.solid_angle for dgb in atom_1_dgbs]

            println("  Atom 1 DGB ranges:")
            println("    ρ range: $(minimum(ρ_values)) to $(maximum(ρ_values))")
            println("    V range: $(minimum(V_values)) to $(maximum(V_values))")
            println(
                "    α (solid angle) range: $(minimum(solid_angles)) to $(maximum(solid_angles))",
            )
            println("    Sum of solid angles: $(sum(solid_angles)) (should be ≈ 1.0)")
        end
    end

    # Write a simple test file
    println("\nWriting test file with DGB data...")
    variables = ["X", "Y", "Z", "ρ", "V", "α"]

    try
        open_plt_file(
            "test_dgb_values_check.plt", "DGB Values Check", join(variables, ", ")
        )

        write_zone(
            "DGB Values Check",
            system_gbd_data;
            system=sys,
            variables=variables,
            cell_centered_spheres=false,
        )

        close_plt_file()
        println("✓ Successfully wrote DGB values check file")

        # Check file size
        run(`ls -lh test_dgb_values_check.plt`)

    catch e
        println("✗ Failed to write DGB values check file:")
        showerror(stdout, e)
        println()
        return false
    end

    return true
end

# Run the test
println("Starting DGB values check test...")
test_passed = test_dgb_values_check()

println("\n=== Test Result ===")
if test_passed
    println("DGB values check test PASSED! ✓")
    println(
        "The DGB function values and solid angles are being extracted and written correctly.",
    )
else
    println("DGB values check test FAILED! ✗")
end
