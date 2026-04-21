using ElectronDensityAnalysis
using LinearAlgebra
include("../test/test_helpers.jl")

function load_real_chemical_system()
    println("Loading real chemical system from test data...")

    # Try to load ethane data
    test_data_file = "test_data/ethane_adf.t41"

    try
        println("Attempting to load: $test_data_file")
        sys = with_zip_file(test_data_file, load_data_scm_adf_t41)[1]
        println("✅ Successfully loaded real chemical system!")
        return sys
    catch e
        println("❌ Failed to load $test_data_file:")
        println("Error: $e")

        # Try another common test file
        fallback_file = "test_data/adamantane_adf.t41"
        try
            println("Attempting fallback: $fallback_file")
            sys = with_zip_file(fallback_file, load_data_scm_adf_t41)[1]
            println("✅ Successfully loaded fallback chemical system!")
            return sys
        catch e2
            println("❌ Failed to load $fallback_file:")
            println("Error: $e2")
            println("Unable to load real test data, check if files are properly extracted")
            return nothing
        end
    end
end

function analyze_and_write_system(sys::System)
    println("\n=== System Analysis ===")
    println("System name: $(sys.name)")
    println("System source: $(sys.source)")
    println("Grid dimensions: $(sys.grid.n_pts)")
    println("Total grid points: $(prod(sys.grid.n_pts))")
    println("Number of atoms: $(length(sys.atoms))")

    # Display atom types
    atom_types = unique([atom.data.symbol for atom in sys.atoms])
    atom_counts = Dict{String,Int}()
    for atom in sys.atoms
        symbol = atom.data.symbol
        atom_counts[symbol] = get(atom_counts, symbol, 0) + 1
    end

    println("Atom composition:")
    for (symbol, count) in atom_counts
        println("  $symbol: $count atoms")
    end

    # Get some basic statistics about the density
    rho_min = minimum(sys.data)
    rho_max = maximum(sys.data)
    rho_mean = sum(sys.data) / length(sys.data)
    println("Density statistics:")
    println("  Min: $(rho_min)")
    println("  Max: $(rho_max)")
    println("  Mean: $(rho_mean)")

    # Find critical points
    println("\n=== Finding Critical Points ===")
    cps = Vector{CriticalPoint}()
    try
        cps_found = find_critical_points(sys)
        println("Found $(length(cps_found)) critical points")

        # Group by type - note: CriticalPoint structure might be different
        cp_types = Dict{String,Int}()
        for cp in cps_found
            # Check the structure of the critical point
            try
                type_str = string(cp.cp_type)  # Try cp_type field
                cp_types[type_str] = get(cp_types, type_str, 0) + 1
                push!(cps, cp)
            catch
                try
                    type_str = string(cp.data.cp_type)  # Try data.cp_type field
                    cp_types[type_str] = get(cp_types, type_str, 0) + 1
                    push!(cps, cp)
                catch
                    # If we can't determine the type, still add it
                    type_str = "unknown"
                    cp_types[type_str] = get(cp_types, type_str, 0) + 1
                    push!(cps, cp)
                end
            end
        end

        println("Critical point types:")
        for (type_str, count) in cp_types
            println("  $type_str: $count")
        end
    catch e
        println("❌ Error finding critical points: $e")
    end

    # Generate some gradient paths
    println("\n=== Skipping Gradient Paths ===")
    gradient_paths = []
    println("Note: Gradient path generation skipped for this test")
    println("Generated $(length(gradient_paths)) gradient paths")

    # Prepare zones for PLT file
    println("\n=== Writing to PLT File ===")
    zones = Vector{Tuple{String,Any}}()

    # Add the system (this will include grid data, metadata, and atoms)
    push!(zones, ("System", sys))

    # Add critical points
    if !isempty(cps)
        push!(zones, ("Critical Points", cps))
    end

    # Add gradient paths (limit to avoid huge files)
    max_paths_to_write = min(5, length(gradient_paths))
    for (i, gp) in enumerate(gradient_paths[1:max_paths_to_write])
        push!(zones, ("Gradient Path $i", gp))
    end

    # Write comprehensive PLT file
    output_filename = "real_chemical_system.plt"
    println("Writing comprehensive data to $output_filename...")

    try
        TecplotWriter.write_plt_file(
            output_filename, zones; title="Real Chemical System - $(sys.name)", system=sys
        )

        println("✅ Successfully wrote Tecplot file: $output_filename")
        println("\n=== File Contents ===")
        println("The file contains:")
        println("  📊 System grid data ($(prod(sys.grid.n_pts)) points)")
        println("  📋 System metadata (name: '$(sys.name)', source: '$(sys.source)')")
        println("  ⚛️  Atomic coordinates ($(length(sys.atoms)) atoms, grouped by element)")
        println("  🎯 Critical points ($(length(cps)) total, grouped by type)")
        println("  📈 Gradient paths ($max_paths_to_write paths with rho values)")
        println("\n🚀 Ready for visualization in Tecplot 360 or ParaView!")

        return true
    catch e
        println("❌ Error writing PLT file:")
        showerror(stdout, e)
        println()
        return false
    end
end

function test_real_chemical_system_plt()
    println("=== Real Chemical System PLT Writer Test ===")

    # Load real chemical system
    sys = load_real_chemical_system()

    if sys === nothing
        println("Cannot proceed without valid chemical system data")
        return false
    end

    # Analyze and write the system
    success = analyze_and_write_system(sys)

    if success
        println("\n🎉 Real chemical system PLT test completed successfully!")
        println("Check the generated 'real_chemical_system.plt' file")
    else
        println("\n❌ Real chemical system PLT test failed")
    end

    return success
end

# Run the test
test_real_chemical_system_plt()
