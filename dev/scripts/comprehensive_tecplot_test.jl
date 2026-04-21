using ElectronDensityAnalysis

# More comprehensive test
function comprehensive_test()
    println("Running comprehensive TecplotWriter test...")

    # Test 1: Just atoms
    println("\n=== Test 1: Atoms ===")
    atoms = [
        NuclearCoordinate([0.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([1.0, 1.0, 1.0], 1),  # Hydrogen
        NuclearCoordinate([2.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([3.0, 1.0, 1.0], 1),  # Hydrogen
    ]

    zones1 = Vector{Tuple{String,Any}}([("Test Atoms", atoms)])
    TecplotWriter.write_plt_file(
        "comprehensive_atoms.plt", zones1; title="Comprehensive Atoms Test"
    )
    println("✓ Created comprehensive_atoms.plt")

    # Test 2: Test with mixed atom types (should create separate zones)
    println("\n=== Test 2: Mixed atom types ===")
    mixed_atoms = [
        NuclearCoordinate([0.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([1.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([2.0, 0.0, 0.0], 1),  # Hydrogen
        NuclearCoordinate([3.0, 0.0, 0.0], 1),  # Hydrogen
        NuclearCoordinate([4.0, 0.0, 0.0], 8),  # Oxygen
    ]

    zones2 = Vector{Tuple{String,Any}}([("Mixed Atoms", mixed_atoms)])
    TecplotWriter.write_plt_file(
        "comprehensive_mixed.plt", zones2; title="Mixed Atom Types Test"
    )
    println("✓ Created comprehensive_mixed.plt")

    println("\nAll comprehensive tests completed successfully!")
    println("Created files:")
    println("  - comprehensive_atoms.plt: Simple atom coordinates")
    println(
        "  - comprehensive_mixed.plt: Mixed atom types (should create separate zones per element)",
    )

    return nothing
end

comprehensive_test()
