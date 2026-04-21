# Make sure the main module is in the path
if !("../src" in LOAD_PATH)
    push!(LOAD_PATH, "../src")
end

using ElectronDensityAnalysis
using .Tecio

function test_tecio_ordered()
    println("Running Tecio IJK-ordered test...")

    # Define grid dimensions
    imax, jmax, kmax = 10, 15, 5
    n_total = imax * jmax * kmax

    # Define variables
    variables = "X, Y, Z, P, T"

    # Generate some data
    x = Float32[i for i in 1:imax for j in 1:jmax for k in 1:kmax]
    y = Float32[j for i in 1:imax for j in 1:jmax for k in 1:kmax]
    z = Float32[k for i in 1:imax for j in 1:jmax for k in 1:kmax]
    p = Float32[i * j * k for i in 1:imax for j in 1:jmax for k in 1:kmax]
    t = rand(Float32, n_total) * 100.0f0

    # Output file
    filename = "ijk_ordered_test.plt"
    scratchdir = "."
    title = "IJK-Ordered Zone Test"

    try
        # Initialize Tecplot file
        Tecio.tecini(title, variables, filename, scratchdir)

        # Add dataset-level auxiliary data
        Tecio.tecauxstr("DataSetAux", "This is a test for ordered data")

        # Write zone header
        Tecio.teczne_ordered("Zone 1", imax, jmax, kmax)

        # Add zone-level auxiliary data
        Tecio.teczauxstr("ZoneAux", "This is a zone test for ordered data")

        # Add variable-level auxiliary data
        for (i, var_name) in enumerate(split(variables, ", "))
            Tecio.tecvauxstr(i, "VarAux_$(var_name)", "Test for $(var_name)")
        end

        # Write data
        Tecio.tecdat(x)
        Tecio.tecdat(y)
        Tecio.tecdat(z)
        Tecio.tecdat(p)
        Tecio.tecdat(t)

        # Close file
        Tecio.tecend()

        println("Successfully wrote Tecplot file: $filename")
        println("Please check the file in a Tecplot viewer.")

    catch e
        println("An error occurred during the Tecio test.")
        showerror(stdout, e)
        println()
    end
end

test_tecio_ordered()
