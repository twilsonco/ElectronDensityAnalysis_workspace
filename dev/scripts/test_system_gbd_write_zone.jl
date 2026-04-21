#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using Serialization: deserialize
include("../test/test_helpers.jl")

# Test comprehensive SystemGBDData write_zone functionality
function test_system_gbd_write_zone()
    println("Testing SystemGBDData write_zone method...")

    # Load the pre-computed system GBD data from checkpoints 
    try
        system_gbd_data = deserialize(open("gbd_checkpoints/system_gbd_data.jls", "r"))
        println("✓ Successfully loaded SystemGBDData from checkpoint")
        println("  - $(length(system_gbd_data.atom_sphere_data)) atom spheres")
        println("  - $(length(system_gbd_data.differential_gradient_bundles)) total DGBs")
        println("  - $(length(system_gbd_data.rejected_gradient_paths)) rejected paths")
        println("  - Function names: $(system_gbd_data.function_names)")

        # Test basic write_zone functionality
        println("\nTesting SystemGBDData write_zone...")

        # Prepare minimal variables (just coordinates)
        variables = ["X", "Y", "Z"]

        # Test 1: Basic write
        try
            open_plt_file(
                "test_system_gbd_basic.plt",
                "SystemGBDData Basic Test",
                join(variables, ", "),
            )
            write_zone("SystemGBD Basic", system_gbd_data)
            close_plt_file()
            println("✓ Basic SystemGBDData write successful")
        catch e
            println("❌ Basic write failed: $e")
            close_plt_file()
        end

        # Test 2: With cell-centered spheres
        try
            open_plt_file(
                "test_system_gbd_cell_centered.plt",
                "SystemGBDData Cell-Centered Test",
                join(variables, ", "),
            )
            write_zone("SystemGBD CC", system_gbd_data; cell_centered_spheres=true)
            close_plt_file()
            println("✓ Cell-centered SystemGBDData write successful")
        catch e
            println("❌ Cell-centered write failed: $e")
            close_plt_file()
        end

        println("✓ All SystemGBDData write_zone tests completed successfully!")

    catch e
        println("❌ Error: $e")
        println("Make sure to run the comprehensive test first to generate checkpoints.")
        return false
    end

    return true
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_system_gbd_write_zone()
end
