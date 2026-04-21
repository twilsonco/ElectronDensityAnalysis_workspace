using ElectronDensityAnalysis
using LinearAlgebra
include("../test/test_helpers.jl")

function test_vector_gradient_paths()
    println("=== Testing Vector{GradientPath} write_zone ===")

    # Load a real system for gradient path analysis
    sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
    println("System: $(sys.name)")

    # Find critical points
    cps = find_critical_points(sys)
    println("Found $(length(cps)) critical points")

    # Create multiple gradient paths from different starting points
    seed_points = [
        [0.0, 0.0, 0.1],   # Near center
        [0.2, 0.0, 0.0],   # Offset in x
        [0.0, 0.2, 0.0],   # Offset in y
    ]

    gradient_paths = GradientPath[]

    for (i, seed_point) in enumerate(seed_points)
        try
            gp = create_gradient_path(seed_point, sys, both_dir; cps=cps)
            push!(gradient_paths, gp)
            println("  Created gradient path $i: $(size(gp.r, 2)) points")
        catch e
            println("  Warning: Could not create gradient path $i from $seed_point: $e")
        end
    end

    println("Successfully created $(length(gradient_paths)) gradient paths")

    # Test Case 1: Write vector of gradient paths using new method
    println("\n--- Test Case 1: Vector{GradientPath} method ---")

    aux_data = [
        "TestCase=Vector GradientPath Method",
        "CreatedBy=TecplotWriter Test",
        "Date=2025-07-31",
    ]

    zones = Tuple{String,Any}[
        ("System", sys),
        ("Critical Points", cps),
        ("Gradient Paths Collection", gradient_paths),  # This will use the new Vector{GradientPath} method
    ]

    TecplotWriter.write_plt_file(
        "test_vector_gradient_paths.plt",
        zones;
        title="Vector{GradientPath} Test - $(sys.name)",
        system=sys,
        aux_data=aux_data,
    )

    println("✅ Vector{GradientPath} PLT file written successfully")

    # Test Case 2: Compare with individual gradient path zones (old way)
    println("\n--- Test Case 2: Individual GradientPath zones (comparison) ---")

    individual_zones = Tuple{String,Any}[("System", sys), ("Critical Points", cps)]

    # Add each gradient path as a separate zone (the old manual way)
    for (i, gp) in enumerate(gradient_paths)
        push!(individual_zones, ("Individual Gradient Path $i", gp))
    end

    TecplotWriter.write_plt_file(
        "test_individual_gradient_paths.plt",
        individual_zones;
        title="Individual GradientPath Test - $(sys.name)",
        system=sys,
        aux_data=aux_data,
    )

    println("✅ Individual GradientPath zones PLT file written successfully")

    # Summary
    println("\n=== Summary ===")
    println("Created two PLT files for comparison:")
    println("  📁 test_vector_gradient_paths.plt - Uses Vector{GradientPath} method")
    println("  📁 test_individual_gradient_paths.plt - Uses individual GradientPath zones")
    println("")
    println("Vector{GradientPath} method benefits:")
    println("  ✅ Cleaner API - one zone entry instead of $(length(gradient_paths))")
    println("  ✅ Consistent naming - automatic numbering")
    println("  ✅ Collection metadata - ParentCollection, PathIndex, TotalPaths aux data")
    println("  ✅ Hierarchical aux_data - inherited from parent with collection context")

    total_points = sum(size(gp.r, 2) for gp in gradient_paths)
    println("\nGradient paths summary:")
    println("  Total paths: $(length(gradient_paths))")
    println("  Total points: $total_points")
    println("")
    println("🎉 Vector{GradientPath} test completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_vector_gradient_paths()
end
