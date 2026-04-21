using ElectronDensityAnalysis

# Test the TecplotWriter functions directly
function simple_test()
    println("Testing TecplotWriter functions directly...")

    # Create some simple test data
    atoms = [
        NuclearCoordinate([0.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([1.0, 1.0, 1.0], 1),  # Hydrogen
    ]

    println("Created $(length(atoms)) test atoms")

    try
        # Try to call the TecplotWriter functions
        zones = Vector{Tuple{String,Any}}([("Test Atoms", atoms)])
        TecplotWriter.write_plt_file("simple_test.plt", zones; title="Simple Test")
        println("✓ Successfully wrote simple_test.plt")
    catch e
        println("✗ Error:")
        showerror(stdout, e)
        println()
    end

    return nothing
end

simple_test()
