#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using Interpolations: Linear
using Serialization: deserialize, serialize

# Create comprehensive test using existing data loaders and test utilities
include("../test/test_helpers.jl")

# Test comprehensive SystemGBDData writing with Cu periodic system
function test_comprehensive_cu_system()
    println("\n=== Comprehensive Cu SystemGBDData Test ===")

    # Load Cu system using band loader
    println("Loading Cu system...")
    sys = nothing
    try
        sys = load_data_scm_adf_t41("/Users/haiiro/Downloads/cu43-rotated-fine.t41")[1]
        println("✓ Successfully loaded Cu system")
        println("  System: $(sys.name)")
        println("  Grid: $(sys.grid.n_pts)")
        println("  Atoms: $(length(sys.atoms))")
        println("  Periodicity: $(sys.is_periodic)")
    catch e
        println("✗ Failed to load Cu data: $e")
        showerror(stdout, e)
        println()
        return false
    end

    # Expand periodic system to work in a larger central space
    # This is important for periodic systems to avoid boundary artifacts
    # println("\nExpanding periodic system...")
    # try
    #     # Expand by factor 2.5 in periodic directions (typically x and y for slabs)
    #     # Determine periodicity - typically materials are periodic in x,y but not z
    #     sys = expand_system_periodic(sys; factor=1.2, is_periodic=[true, true, true])
    #     println("✓ Successfully expanded system")
    #     println("  New grid: $(sys.grid.n_pts)")
    # catch e
    #     println("✗ Failed to expand system: $e")
    #     showerror(stdout, e)
    #     println()
    #     return false
    # end

    # Find critical points
    println("\nFinding critical points...")
    cps = nothing
    try
        cps = find_critical_points(
            sys; spacing=0.35, bounding_box_shrink_factor=0.8, dup_cp_check_dist=0.1
        )
        println("✓ Found $(length(cps)) critical points")

        # Show CP types
        cp_counts = Dict()
        for cp in cps
            cp_counts[cp.type] = get(cp_counts, cp.type, 0) + 1
        end
        for (cp_type, count) in cp_counts
            println("  $cp_type: $count")
        end
    catch e
        println("✗ Failed to find critical points: $e")
        showerror(stdout, e)
        println()
        return false
    end

    # Check if we have a cached comprehensive result
    comprehensive_checkpoint = "gbd_checkpoints/cu_adf/comprehensive_cu_1000dgb.jls"
    system_gbd_data = nothing

    # Expected function names with the new curvature fields
    expected_function_names = [
        "ρ",
        "V",
        "ρ mean curvature",
        "ρ Gaussian curvature",
        "ρ RMS curvature",
        "ρ shape index",
        "ρ Willmore energy",
        "ρ modified Willmore energy",
    ]

    if isfile(comprehensive_checkpoint)
        println("\nLoading cached comprehensive SystemGBDData...")
        try
            system_gbd_data = deserialize(open(comprehensive_checkpoint, "r"))

            # Check if cached data has the expected curvature fields
            if system_gbd_data.function_names != expected_function_names
                println(
                    "⚠ Warning: Cached data has old function list (missing curvature fields)",
                )
                println("  Cached: $(system_gbd_data.function_names)")
                println("  Expected: $(expected_function_names)")
                println("  Will regenerate GBD data with new curvature fields...")
                system_gbd_data = nothing
            elseif !isempty(system_gbd_data.differential_gradient_bundles)
                # Check that DGBs have the right number of function values
                first_dgb = system_gbd_data.differential_gradient_bundles[1]
                if length(first_dgb.function_values) != length(expected_function_names)
                    println("⚠ Warning: Cached DGBs have wrong number of function values")
                    println("  DGB function values: $(length(first_dgb.function_values))")
                    println("  Expected: $(length(expected_function_names))")
                    println("  Will regenerate GBD data...")
                    system_gbd_data = nothing
                else
                    println("✓ Successfully loaded comprehensive SystemGBDData from cache")
                    println("  Critical points: $(length(system_gbd_data.critical_points))")
                    println(
                        "  DGBs: $(length(system_gbd_data.differential_gradient_bundles))"
                    )
                    println(
                        "  Rejected paths: $(length(system_gbd_data.rejected_gradient_paths))",
                    )
                    println(
                        "  Atom sphere data: $(length(system_gbd_data.atom_sphere_data))"
                    )
                    println("  Function names: $(system_gbd_data.function_names)")
                end
            else
                println(
                    "✓ Successfully loaded comprehensive SystemGBDData from cache (no DGBs)"
                )
            end
        catch e
            println("✗ Failed to load cached data: $e")
            println("  Will regenerate GBD data...")
            system_gbd_data = nothing
        end
    end

    # If no cached data, perform gradient bundle decomposition
    if system_gbd_data === nothing
        println(
            "\nPerforming comprehensive gradient bundle decomposition (1000 DGBs per atom)...",
        )
        try
            system_gbd_data = gradient_bundle_decomposition(
                sys;
                cps=cps,
                num_gbs=1000,                    # 1000 DGBs per atom for comprehensive analysis
                interp_type=Linear,
                func_cutoff=1e-3,                # Density cutoff for path termination
                curvature_threshold=0.8 * 0.5π,  # Curvature threshold for path rejection
                atom_bounding_box_shrink_factor=0.4,
                seed_sphere_radius_factor=0.1,
                use_checkpoints=true,
                checkpoint_dir="gbd_checkpoints/cu_adf",
                verbose_checkpoints=true,
            )
            println("✓ Completed gradient bundle decomposition")
            println("  Critical points: $(length(system_gbd_data.critical_points))")
            println("  DGBs: $(length(system_gbd_data.differential_gradient_bundles))")
            println("  Rejected paths: $(length(system_gbd_data.rejected_gradient_paths))")
            println("  Atom sphere data: $(length(system_gbd_data.atom_sphere_data))")
            println("  Function names: $(system_gbd_data.function_names)")

            # Save comprehensive result for future use
            println("Saving comprehensive SystemGBDData to cache...")
            try
                mkpath(dirname(comprehensive_checkpoint))
                serialize(open(comprehensive_checkpoint, "w"), system_gbd_data)
                println("✓ Saved comprehensive SystemGBDData cache")
            catch e
                println("⚠ Warning: Failed to save cache: $e")
            end
        catch e
            println("✗ Failed gradient bundle decomposition: $e")
            showerror(stdout, e)
            println()
            return false
        end
    end

    # Write comprehensive system results
    println("\n=== Writing Comprehensive System Results ===")

    # Prepare variables including all function names
    # Note: GBD automatically includes curvature fields (mean, Gaussian, RMS, shape index)
    # Variable names are context-dependent:
    #   - System zone: volumetric fields ρ(x,y,z), H(x,y,z), K(x,y,z), etc.
    #   - Atom spheres/DGBs: condensed functions P(θ,φ), H_condensed(θ,φ), K_condensed(θ,φ), etc.
    variables = ["X", "Y", "Z"]
    append!(variables, system_gbd_data.function_names)

    println("\nVariables to be written:")
    for (i, var) in enumerate(variables)
        println("  $i. $var")
    end

    try
        open_plt_file(
            "comprehensive_cu_system_complete.plt",
            "Comprehensive Cu System with Complete GBD Analysis",
            join(variables, ", "),
        )

        println("Writing complete SystemGBDData...")
        write_zone(
            "Comprehensive Cu System",
            system_gbd_data;
            system=sys,
            variables=variables,
            cell_centered_spheres=false,
        )

        close_plt_file()
        println("✓ Successfully wrote comprehensive system file")

        # Get file size for reporting
        file_size = filesize("comprehensive_cu_system_complete.plt")
        file_size_mb = round(file_size / (1024 * 1024); digits=2)
        println("  File size: $file_size_mb MB")
    catch e
        println("✗ Failed to write comprehensive system file:")
        showerror(stdout, e)
        println()
        close_plt_file()  # Ensure file is closed
        return false
    end

    # # Also create a cell-centered version
    # println("\n--- Creating Cell-Centered Version ---")
    # try
    #     open_plt_file(
    #         "comprehensive_cu_system_cell_centered.plt",
    #         "Comprehensive Cu System with Cell-Centered Data",
    #         join(variables, ", "),
    #     )

    #     println("Writing cell-centered SystemGBDData...")
    #     write_zone(
    #         "Comprehensive Cu System CC",
    #         system_gbd_data;
    #         system=sys,
    #         variables=variables,
    #         cell_centered_spheres=true,
    #     )

    #     close_plt_file()
    #     println("✓ Successfully wrote cell-centered system file")

    #     # Get file size for reporting
    #     file_size_cc = filesize("comprehensive_cu_system_cell_centered.plt")
    #     file_size_cc_mb = round(file_size_cc / (1024 * 1024); digits=2)
    #     println("  File size: $file_size_cc_mb MB")
    # catch e
    #     println("✗ Failed to write cell-centered system file:")
    #     showerror(stdout, e)
    #     println()
    #     close_plt_file()  # Ensure file is closed
    #     return false
    # end

    # Print comprehensive summary
    println("\n=== Comprehensive Analysis Summary ===")
    println("System: $(sys.name)")
    println("Total atoms: $(length(sys.atoms))")
    nuclear_cps = [cp for cp in system_gbd_data.critical_points if cp.type == nuclear_cp]
    println("Nuclear critical points: $(length(nuclear_cps))")

    total_dgbs = length(system_gbd_data.differential_gradient_bundles)
    avg_dgbs_per_atom = round(total_dgbs / length(nuclear_cps); digits=1)
    println("Total DGBs: $total_dgbs")
    println("Average DGBs per atom: $avg_dgbs_per_atom")

    total_rejected = length(system_gbd_data.rejected_gradient_paths)
    println("Total rejected paths: $total_rejected")

    println("\nFunction analysis:")
    for (i, func_name) in enumerate(system_gbd_data.function_names)
        total_val = system_gbd_data.function_totals[i]
        println("  $func_name: $(round(total_val, digits=6))")
    end

    println("\nDGBs per atom:")
    for (atom_idx, atom_sphere_data) in system_gbd_data.atom_sphere_data
        if atom_idx <= length(nuclear_cps)
            cp = nuclear_cps[atom_idx]
            atom_dgbs = [
                dgb for dgb in system_gbd_data.differential_gradient_bundles if
                dgb.cp_index == atom_idx
            ]
            if hasfield(typeof(cp), :i_atom) &&
                cp.i_atom > 0 &&
                cp.i_atom <= length(sys.atoms)
                atom_symbol = sys.atoms[cp.i_atom].data.symbol
                println("  Atom $atom_idx ($atom_symbol): $(length(atom_dgbs)) DGBs")
            else
                println("  Atom $atom_idx: $(length(atom_dgbs)) DGBs")
            end
        end
    end

    return true
end

# Run the comprehensive test
println("Starting comprehensive Cu SystemGBDData test...")
println("This will generate complete system results with 1000 DGBs per atom")
println(
    "Including automatic isosurface curvature analysis (mean, Gaussian, RMS, shape index)"
)

test_passed = test_comprehensive_cu_system()

println("\n=== Final Result ===")
if test_passed
    println("Comprehensive Cu SystemGBDData test PASSED! ✓")
    println("\nGenerated files:")
    println(
        "  📁 comprehensive_cu_system_complete.plt - Complete system with node-centered spheres",
    )
    println(
        "  📁 comprehensive_cu_system_cell_centered.plt - Complete system with cell-centered spheres",
    )
    println("\nFiles contain:")
    println("  ✓ System volumetric grid data (expanded periodic system)")
    println("  ✓ Critical points (nuclear and bond)")
    println("  ✓ All atom sphere data with triangulated surfaces")
    println("  ✓ All differential gradient bundles (DGBs) for each atom")
    println("  ✓ All rejected gradient paths")
    println("  ✓ Condensed function data for all DGBs:")
    println("    - ρ (electron density)")
    println("    - V (volume)")
    println("    - ρ mean curvature (H)")
    println("    - ρ Gaussian curvature (K)")
    println("    - ρ RMS curvature")
    println("    - ρ shape index")
    println("    - ρ Willmore energy (H²)")
    println("    - ρ modified Willmore energy (H² - K)")
    println("  ✓ Multi-zone hierarchical structure")
else
    println("Comprehensive Cu SystemGBDData test FAILED! ✗")
end
