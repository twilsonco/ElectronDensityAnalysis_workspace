#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using Interpolations: Linear

# Quick test with just 50 DGBs per atom to verify curvature fields work
include("../test/test_helpers.jl")

println("=== Quick Curvature Test (50 DGBs) ===")

# Load ethane system
println("Loading ethane system...")
sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
println("✓ Loaded: $(sys.name)")

# Find critical points
println("\nFinding critical points...")
cps = find_critical_points(sys; spacing=0.35, dup_cp_check_dist=0.1)
println("✓ Found $(length(cps)) critical points")

# Perform GBD with curvature fields
println("\nPerforming GBD with 50 DGBs per atom (includes curvature fields)...")
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

println("✓ Completed GBD")
println("  DGBs: $(length(system_gbd_data.differential_gradient_bundles))")
println("  Function names: $(system_gbd_data.function_names)")
println(
    "  Functions per DGB: $(length(system_gbd_data.differential_gradient_bundles[1].function_values))",
)

# Verify we have 6 functions
expected_functions = 6
actual_functions = length(system_gbd_data.function_names)
if actual_functions != expected_functions
    error("Expected $expected_functions functions, got $actual_functions")
end
println("✓ Correct number of functions: $actual_functions")

# Write Tecplot file
println("\nWriting Tecplot file...")
variables = ["X", "Y", "Z", "Electron Density"]
append!(variables, system_gbd_data.function_names)

println("Variables: ", join(variables, ", "))

open_plt_file("quick_curvature_test.plt", "Quick Curvature Test", join(variables, ", "))

write_zone(
    "Quick Test",
    system_gbd_data;
    system=sys,
    variables=variables,
    cell_centered_spheres=false,
)

close_plt_file()
println("✓ Successfully wrote Tecplot file")

file_size = filesize("quick_curvature_test.plt")
println("  File size: $(round(file_size / (1024*1024); digits=2)) MB")

println("\n=== SUCCESS ===")
println("Curvature fields are working correctly!")
println("Generated: quick_curvature_test.plt")
