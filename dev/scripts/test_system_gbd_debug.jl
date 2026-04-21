#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using Serialization: deserialize
include("../test/test_helpers.jl")

# Test comprehensive SystemGBDData write success by examining what parts work
function test_system_gbd_debug()
    println("Testing SystemGBDData write functionality with debugging...")

    # Load pre-computed system GBD data
    try
        system_gbd_data = deserialize(open("gbd_checkpoints/system_gbd_data.jls", "r"))
        println("✓ Successfully loaded SystemGBDData from checkpoint")
        println("  - System: $(system_gbd_data.system.name)")
        println("  - $(length(system_gbd_data.atom_sphere_data)) atom spheres")
        println("  - $(length(system_gbd_data.differential_gradient_bundles)) total DGBs")
        println("  - Function names: $(system_gbd_data.function_names)")

        # Test individual components first
        println("\n=== Testing individual components ===")

        # Test 1: Just Critical Points
        try
            println("Testing critical points zone...")
            variables = ["X", "Y", "Z"]
            open_plt_file(
                "test_debug_critical_points.plt",
                "Debug Critical Points",
                join(variables, ", "),
            )
            write_zone(
                "Critical Points",
                system_gbd_data.critical_points;
                system=system_gbd_data.system,
                variables=variables,
            )
            close_plt_file()
            println("✓ Critical points write successful")
        catch e
            println("❌ Critical points write failed: $e")
            close_plt_file()
        end

        # Test 2: Just atom spheres - let's try one atom
        try
            println("Testing single atom sphere...")
            variables = ["X", "Y", "Z", "ρ", "V"]  # Include function names

            # Get first atom sphere data
            first_cp_index = first(keys(system_gbd_data.atom_sphere_data))
            first_atom_sphere = system_gbd_data.atom_sphere_data[first_cp_index]

            open_plt_file(
                "test_debug_atom_sphere.plt", "Debug Atom Sphere", join(variables, ", ")
            )
            write_zone(
                "Atom Sphere",
                first_atom_sphere;
                system=system_gbd_data.system,
                variables=variables,
            )
            close_plt_file()
            println("✓ Atom sphere write successful")
        catch e
            println("❌ Atom sphere write failed: $e")
            close_plt_file()
        end

        # Test 3: Just DGBs for one atom - try individual DGBs
        try
            println("Testing individual DGBs...")
            variables = ["X", "Y", "Z", "ρ", "V"]

            # Get DGBs for first atom
            first_cp_index = first(keys(system_gbd_data.atom_sphere_data))
            atom_dgbs = [
                dgb for dgb in system_gbd_data.differential_gradient_bundles if
                dgb.cp_index == first_cp_index
            ]

            open_plt_file("test_debug_dgbs.plt", "Debug DGBs", join(variables, ", "))

            # Write first few individual DGBs instead of using the vector method
            for (i, dgb) in enumerate(atom_dgbs[1:min(3, length(atom_dgbs))])
                write_zone(
                    "DGB $i",
                    dgb.gradient_path;
                    system=system_gbd_data.system,
                    variables=variables,
                )
            end

            close_plt_file()
            println(
                "✓ Individual DGBs write successful ($(min(3, length(atom_dgbs))) DGBs)"
            )
        catch e
            println("❌ Individual DGBs write failed: $e")
            close_plt_file()
        end

        println("\n✅ Component testing completed!")

    catch e
        println("❌ Error: $e")
        return false
    end

    return true
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_system_gbd_debug()
end
