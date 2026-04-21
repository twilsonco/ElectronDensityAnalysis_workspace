# Make sure the main module is in the path
if !("../src" in LOAD_PATH)
    push!(LOAD_PATH, "../src")
end

using ElectronDensityAnalysis
using .Tecio

function test_tecio_fe_tetra()
    println("Running Tecio FE Tetrahedral test...")

    # Define mesh properties
    nnodes = 5
    ncells = 2
    n_total_nodes = nnodes * 1 # For node-centered data

    # Define variables
    variables = "X, Y, Z, T"

    # Generate some data
    # Node coordinates
    x = Float32[0.0, 1.0, 1.0, 0.0, 0.5]
    y = Float32[0.0, 0.0, 1.0, 1.0, 0.5]
    z = Float32[0.0, 0.0, 0.0, 0.0, 1.0]
    # Temperature at nodes
    t = Float32[10.0, 20.0, 30.0, 40.0, 50.0]

    # Connectivity (2 tetrahedra)
    # Tecplot uses 1-based indexing for connectivity
    connectivity = Int32[
        1,
        2,
        4,
        5,  # First tetrahedron
        2,
        3,
        4,
        5,   # Second tetrahedron
    ]

    # Output file
    filename = "fe_tetra_test.plt"
    scratchdir = "."
    title = "FE Tetrahedral Zone Test"

    try
        # Initialize Tecplot file
        Tecio.tecini(title, variables, filename, scratchdir)

        # Add dataset-level auxiliary data
        Tecio.tecauxstr("DataSetAux", "This is a test")

        # Write zone header
        Tecio.teczne_fe_tetra("Zone 1", nnodes, ncells)

        # Add zone-level auxiliary data
        Tecio.teczauxstr("ZoneAux", "This is a zone test")

        # Add variable-level auxiliary data
        # This must be done for each variable after the zone header
        for (i, var_name) in enumerate(split(variables, ", "))
            Tecio.tecvauxstr(i, "VarAux_$(var_name)", "Test for $(var_name)")
        end

        # Write data (node-centered)
        Tecio.tecdat(x)
        Tecio.tecdat(y)
        Tecio.tecdat(z)
        Tecio.tecdat(t)

        # Write connectivity
        Tecio.tecnode(connectivity)

        # Close file
        Tecio.tecend()

        println("Successfully wrote Tecplot file: $filename")
        println("Please check the file in a Tecplot viewer.")

    catch e
        println("An error occurred during the Tecio FE test.")
        showerror(stdout, e)
        println()
    end
end

test_tecio_fe_tetra()
