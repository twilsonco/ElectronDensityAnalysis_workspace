using ElectronDensityAnalysis
using LinearAlgebra
using Random
using Interpolations
include("../test/test_helpers.jl")

# Import TecplotWriter from main module if it's included there, 
# otherwise include it directly
if isdefined(ElectronDensityAnalysis, :TecplotWriter)
    using ElectronDensityAnalysis.TecplotWriter
else
    include("../src/io/TecplotWriter.jl")
    using .TecplotWriter
end

function generate_seed_points(sys::System, N::Int, spread::Float64=4.0)
    T = eltype(sys.grid.origin)
    center = sys.grid.origin + sys.grid.lattice_full * 0.5ones(3)
    seeds = Matrix{T}(rand(3, N) .- 0.5)  # Random values between -0.5 and 0.5
    seeds .*= spread  # Scale the spread
    seeds .+= center  # Center around the middle of the grid
    bb_gs = shrink_grid_bounds(sys.grid, sys.is_periodic, 0.2)
    clamp_to_bounding_box!(seeds, bb_gs)
    return seeds
end

function create_synthetic_system()
    # Create a simple synthetic system for testing
    println("Creating synthetic test system...")

    # Create a simple 3D grid
    origin = [0.0, 0.0, 0.0]
    lattice = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    n_pts = [10, 10, 10]

    gs = GridSpec(origin, lattice, n_pts)

    # Create some synthetic density data (a simple Gaussian)
    data = zeros(Float32, n_pts...)
    center = [5.0, 5.0, 5.0]
    for i in 1:n_pts[1], j in 1:n_pts[2], k in 1:n_pts[3]
        pos = [i-1, j-1, k-1] .* 0.1
        r = norm(pos - center .* 0.1)
        data[i, j, k] = exp(-r^2 / 0.5)
    end

    # Create some atoms
    atoms = [
        NuclearCoordinate([0.25, 0.25, 0.25], 6),  # Carbon
        NuclearCoordinate([0.75, 0.75, 0.75], 1),  # Hydrogen
        NuclearCoordinate([0.25, 0.75, 0.25], 6),  # Carbon
        NuclearCoordinate([0.75, 0.25, 0.75], 1),  # Hydrogen
    ]

    # Create interpolation function
    g = generate_interpolation_grid(gs)
    itp = interpolate((g[1], g[2], g[3]), data, Gridded(Linear()))
    func = (r) -> itp(r...)

    # Create system
    sys = System(
        "Synthetic Test System",
        "TecplotWriter Test",
        gs,
        data,
        func,
        atoms,
        [false, false, false],  # not periodic
    )

    return sys
end

function test_tecplot_writer()
    println("Testing TecplotWriter with various data types...")

    # Create some test atoms with different elements
    atoms = [
        NuclearCoordinate([0.0, 0.0, 0.0], 6),   # Carbon
        NuclearCoordinate([1.0, 0.0, 0.0], 6),   # Carbon
        NuclearCoordinate([2.0, 0.0, 0.0], 1),   # Hydrogen
        NuclearCoordinate([3.0, 0.0, 0.0], 1),   # Hydrogen
        NuclearCoordinate([0.0, 1.0, 0.0], 8),   # Oxygen
        NuclearCoordinate([1.0, 1.0, 0.0], 7),   # Nitrogen
    ]

    println("Created $(length(atoms)) test atoms")
    println("Atom types: Carbon, Hydrogen, Oxygen, Nitrogen")

    # Prepare data for writing to PLT file
    zones = Vector{Tuple{String,Any}}()

    # Add atoms (this should create separate zones per element type)
    push!(zones, ("Test Atoms", atoms))

    # Write to PLT file
    output_filename = "tecplot_writer_test.plt"
    println("Writing data to $output_filename...")

    try
        TecplotWriter.write_plt_file(
            output_filename, zones; title="ElectronDensityAnalysis Test Data"
        )
        println("Successfully wrote Tecplot file: $output_filename")
        println("The file should contain:")
        println("  - All atoms in one zone")
        println("  - Separate zones for each element type (C, H, O, N)")
        println("  - Atomic coordinates (X, Y, Z)")
        println("  - Atomic numbers")
        println()
        println(
            "You can open this file in Tecplot 360 or ParaView to visualize the results."
        )

    catch e
        println("An error occurred while writing the Tecplot file:")
        showerror(stdout, e)
        println()
    end

    return nothing
end

function test_individual_components()
    println("Testing individual TecplotWriter components...")

    # Test 1: Simple atoms
    println("\nTest 1: Writing simple atoms...")
    simple_atoms = [
        NuclearCoordinate([0.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([1.0, 1.0, 1.0], 1),  # Hydrogen
    ]

    zones1 = Vector{Tuple{String,Any}}([("Simple Atoms", simple_atoms)])
    TecplotWriter.write_plt_file("test_simple_atoms.plt", zones1; title="Simple Atoms Test")

    # Test 2: Mixed element types
    println("Test 2: Writing mixed element types...")
    mixed_atoms = [
        NuclearCoordinate([0.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([1.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([2.0, 0.0, 0.0], 1),  # Hydrogen
        NuclearCoordinate([0.0, 1.0, 0.0], 8),  # Oxygen
    ]

    zones2 = Vector{Tuple{String,Any}}([("Mixed Atoms", mixed_atoms)])
    TecplotWriter.write_plt_file("test_mixed_atoms.plt", zones2; title="Mixed Atoms Test")

    println("Individual component tests completed!")
    println("Created files:")
    println("  - test_simple_atoms.plt: Simple C and H atoms")
    println("  - test_mixed_atoms.plt: Mixed C, H, and O atoms")
end

function test_minimal_tecplot_writer()
    println("Testing TecplotWriter with minimal synthetic data...")

    # Create some test atoms
    atoms = [
        NuclearCoordinate([0.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([1.0, 1.0, 1.0], 1),  # Hydrogen
        NuclearCoordinate([2.0, 0.0, 0.0], 6),  # Carbon
        NuclearCoordinate([3.0, 1.0, 1.0], 1),  # Hydrogen
    ]

    # Test writing atoms only
    println("Test 1: Writing atoms to PLT file...")
    try
        write_plt_file(
            "test_atoms_minimal.plt", [("Test Atoms", atoms)]; title="Minimal Atoms Test"
        )
        println("✓ Successfully wrote atoms-only PLT file: test_atoms_minimal.plt")
    catch e
        println("✗ Error writing atoms PLT file:")
        showerror(stdout, e)
        println()
    end

    println("Minimal test completed!")
end

# Run the comprehensive tests
test_tecplot_writer()

# Run individual component tests
test_individual_components()
