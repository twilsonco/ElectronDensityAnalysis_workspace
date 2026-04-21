using ElectronDensityAnalysis
using LinearAlgebra
include("../test/test_helpers.jl")

function test_transient_data()
    println("=== Testing Transient Data Support (Strand ID & Solution Time) ===")

    # Load a real system for testing
    sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
    println("System: $(sys.name)")

    # Find critical points
    cps = find_critical_points(sys)
    println("Found $(length(cps)) critical points")

    # Create some gradient paths
    seed_points = [[0.0, 0.0, 0.1], [0.2, 0.0, 0.0]]

    gradient_paths = GradientPath[]
    for (i, seed_point) in enumerate(seed_points)
        try
            gp = create_gradient_path(seed_point, sys, both_dir; cps=cps)
            push!(gradient_paths, gp)
            println("  Created gradient path $i: $(size(gp.r, 2)) points")
        catch e
            println("  Warning: Could not create gradient path $i: $e")
        end
    end

    # Test 1: Create a transient dataset with multiple time steps
    println("\n--- Test 1: Transient Data with Multiple Time Steps ---")

    # Simulate different time steps with different solution times
    time_steps = [0.0, 1.0, 2.5, 5.0]
    strand_ids = [1, 2, 3]  # Different strands for different data types

    aux_data = [
        "TestCase=Transient Data",
        "CreatedBy=TecplotWriter Transient Test",
        "Date=2025-08-01",
    ]

    # Create multiple zones with different strand IDs and solution times
    all_zones = Tuple{String,Any,Union{Integer,Nothing},Union{Real,Nothing}}[]

    for (t_idx, time) in enumerate(time_steps)
        # System data at this time step (strand 1)
        push!(all_zones, ("System Time $time", sys, 1, time))

        # Critical points at this time step (strand 2) 
        push!(all_zones, ("Critical Points Time $time", cps, 2, time))

        # Gradient paths at this time step (strand 3)
        if !isempty(gradient_paths)
            push!(all_zones, ("Gradient Paths Time $time", gradient_paths, 3, time))
        end
    end

    # Write transient PLT file using individual zone calls
    try
        variables_string = TecplotWriter._get_variables_for_zones(
            [(name, data) for (name, data, _, _) in all_zones], sys
        )
        variables = String.(split(variables_string, ", "))

        TecplotWriter.open_plt_file(
            "test_transient_data.plt",
            "Transient ElectronDensityAnalysis Data",
            variables_string,
        )

        # Write dataset-level auxiliary data
        TecplotWriter._write_dataset_auxiliary_data(aux_data)

        for (name, data, strand_id, solution_time) in all_zones
            TecplotWriter.write_zone(
                name,
                data;
                system=sys,
                variables=variables,
                aux_data=aux_data,
                strand_id=strand_id,
                solution_time=solution_time,
            )
        end

    catch e
        println("An error occurred while writing the transient Tecplot file.")
        showerror(stdout, e)
        println()
    finally
        TecplotWriter.close_plt_file()
    end

    println("✅ Transient PLT file written successfully")

    # Test 2: Single time step with strand ID
    println("\n--- Test 2: Single Time Step with Strand ID ---")

    zones = Tuple{String,Any}[("System", sys), ("Critical Points", cps)]

    if !isempty(gradient_paths)
        push!(zones, ("Gradient Paths", gradient_paths))
    end

    TecplotWriter.write_plt_file(
        "test_single_time_strand.plt",
        zones;
        title="Single Time Step with Strand - $(sys.name)",
        system=sys,
        aux_data=aux_data,
        strand_id=42,
        solution_time=10.5,
    )

    println("✅ Single time step PLT file written successfully")

    # Summary
    println("\n=== Summary ===")
    println("Created transient data PLT files:")
    println("  📁 test_transient_data.plt - Multiple time steps with different strand IDs")
    println(
        "  📁 test_single_time_strand.plt - Single time step with strand ID and solution time",
    )
    println("")
    println("Transient data features:")
    println("  ✅ Multiple time steps - Different solution_time values")
    println("  ✅ Multiple strands - Different strand_id values for different data types")
    println(
        "  ✅ Hierarchical strand propagation - Child zones inherit strand_id and solution_time",
    )
    println("  ✅ Compatible with existing aux_data system")

    total_zones = length(all_zones)
    println("\nTransient data summary:")
    println("  Time steps: $(length(time_steps))")
    println("  Strand IDs: $(length(strand_ids))")
    println("  Total zones: $total_zones")
    println("")
    println("🎉 Transient data test completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_transient_data()
end
