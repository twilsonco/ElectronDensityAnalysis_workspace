using ElectronDensityAnalysis
using LinearAlgebra
include("../test/test_helpers.jl")

function verify_plt_variables()
    println("=== PLT File Variable Verification ===")

    # Load real system
    real_sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
    println("Loaded system: $(real_sys.name)")
    println("Grid dimensions: $(real_sys.grid.n_pts)")

    # Find critical points
    cps = find_critical_points(real_sys)
    println("Found $(length(cps)) critical points")

    # Create zones
    zones = [("System", real_sys), ("Critical Points", cps)]

    # Check what variables would be generated
    variables_string = ElectronDensityAnalysis.TecplotWriter._get_variables_for_zones(
        zones, real_sys
    )
    println("\n=== Variables in PLT file ===")
    println("Full variables string: '$variables_string'")

    # Parse variables correctly (comma-separated with spaces)
    variables = [strip(v) for v in split(variables_string, ",")]
    for (i, var) in enumerate(variables)
        println("$i. '$var'")
    end

    # Write test file 
    println("\n=== Writing verification file ===")
    TecplotWriter.write_plt_file(
        "variables_verification.plt",
        zones;
        title="Variable Verification - $(real_sys.name)",
        system=real_sys,
    )

    println("✅ Variables verification complete!")
    println("Variables in file: $variables_string")

    # Check for the specific issue mentioned
    if "rho" in variables
        println("✅ 'rho' variable present")
    else
        println("❌ 'rho' variable missing")
    end

    # Check that we don't have duplicate density variables
    density_vars = filter(v -> occursin("SCF Density", v) || v == "rho", variables)
    println("Density-related variables: $density_vars")

    if length(density_vars) == 1 && density_vars[1] == "rho"
        println("✅ Only one density variable ('rho') - no duplicates!")
    else
        println("❌ Multiple density variables detected: $density_vars")
    end

    println("\n🎉 Verification complete! Check 'variables_verification.plt'")
end

# Run verification
verify_plt_variables()
