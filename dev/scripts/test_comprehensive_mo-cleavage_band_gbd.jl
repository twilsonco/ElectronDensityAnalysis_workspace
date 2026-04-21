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
function test_comprehensive_band_system(input_path::String="")
    input_path_without_ext = replace(input_path, r"\.[^.]*$" => "")

    if isempty(input_path)
        println("No input path provided. Quitting test.")
        return false
    end

    # Load Cu system using band loader
    println("Loading system...")
    sys = nothing
    try
        sys = load_data_scm_band_t41(input_path; load_var="rho")
        println("✓ Successfully loaded system")
        println("  System: $(sys.name)")
        println("  Grid: $(sys.grid.n_pts)")
        println("  Atoms: $(length(sys.atoms))")
        println("  Periodicity: $(sys.is_periodic)")
    catch e
        println("✗ Failed to load system: $e")
        showerror(stdout, e)
        println()
        return false
    end

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
    comprehensive_checkpoint = "$(input_path_without_ext)/comprehensive_checkpoint.jls"
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
            if !all(name -> name in system_gbd_data.function_names, expected_function_names)
                println(
                    "⚠ Warning: Cached data has old function list (missing curvature fields)",
                )
                println("  Cached: $(system_gbd_data.function_names)")
                println("  Expected: $(expected_function_names)")
                println("  Will regenerate GBD data with new curvature fields...")
                system_gbd_data = nothing
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

    input_path_dir = dirname(input_path)

    # If no cached data, perform gradient bundle decomposition
    if system_gbd_data === nothing
        println("\nPerforming comprehensive gradient bundle decomposition...")
        try
            system_gbd_data = gradient_bundle_decomposition(
                sys;
                cps=cps,
                num_gbs=20000,
                interp_type=Linear,
                func_cutoff=1e-3,                # Density cutoff for path termination
                curvature_threshold=0.8 * 0.5π,  # Curvature threshold for path rejection
                atom_bounding_box_shrink_factor=0.4,
                seed_sphere_radius_factor=0.25,
                use_checkpoints=true,
                checkpoint_dir="$(input_path_without_ext)/gbd_checkpoints",
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

    println("\n=== Extracting simplified spherical meshes for atom spheres ===")
    # For each atom sphere, extract simplified spherical mesh for Tecplot writing
    for (cp_index, atom_data) in system_gbd_data.atom_sphere_data
        println("Processing Atom sphere for cp index $cp_index...")

        # Loop over all functions
        for (func_idx, func_name) in enumerate(system_gbd_data.function_names)
            # Skip all variables except volume "V".
            if func_name != "V"
                println("  Skipping function $func_name (not volume)")
                continue
            end

            try
                spherical_mesh = extract_spherical_mesh(
                    system_gbd_data, cp_index; function_idx=func_idx
                )
                println("✓ Extracted simplified spherical mesh for function: $func_name")
                # Save spherical mesh to serialized file
                # Sanitize function name for filename (replace spaces and special chars)
                safe_func_name = replace(func_name, " " => "_", "/" => "_", "\\" => "_")

                mesh_filename = "$(input_path_without_ext)_atom_$(cp_index)_$(safe_func_name)_spherical_mesh.jls"
                try
                    mkpath(dirname(mesh_filename))
                    serialize(open(mesh_filename, "w"), spherical_mesh)
                    println("✓ Saved spherical mesh to $mesh_filename")
                catch e
                    println("⚠ Warning: Failed to save spherical mesh: $e")
                end
            catch e
                println(
                    "✗ Failed to extract spherical mesh for Atom sphere cp index $cp_index, function $func_name: $e",
                )
                return false
            end
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
            "$input_path_without_ext.plt",
            "GBD Analysis of $(sys.name)",
            join(variables, ", "),
        )

        println("Writing complete SystemGBDData...")
        write_zone(
            "$(basename(input_path_without_ext))_complete_band",
            system_gbd_data;
            system=sys,
            variables=variables,
            cell_centered_spheres=false,
            print_dgbs=false,
        )

        close_plt_file()
        println("✓ Successfully wrote comprehensive system file")

        # Get file size for reporting
        file_size = filesize("$(input_path_without_ext).plt")
        file_size_mb = round(file_size / (1024 * 1024); digits=2)
        println("  File size: $file_size_mb MB")
    catch e
        println("✗ Failed to write comprehensive system file:")
        showerror(stdout, e)
        println()
        close_plt_file()  # Ensure file is closed
        return false
    end

    # # Print comprehensive summary
    # println("\n=== Comprehensive Analysis Summary ===")
    # println("System: $(sys.name)")
    # println("Total atoms: $(length(sys.atoms))")
    # nuclear_cps = [cp for cp in system_gbd_data.critical_points if cp.type == nuclear_cp]
    # println("Nuclear critical points: $(length(nuclear_cps))")

    # total_dgbs = length(system_gbd_data.differential_gradient_bundles)
    # avg_dgbs_per_atom = round(total_dgbs / length(nuclear_cps); digits=1)
    # println("Total DGBs: $total_dgbs")
    # println("Average DGBs per atom: $avg_dgbs_per_atom")

    # total_rejected = length(system_gbd_data.rejected_gradient_paths)
    # println("Total rejected paths: $total_rejected")

    # println("\nFunction analysis:")
    # for (i, func_name) in enumerate(system_gbd_data.function_names)
    #     total_val = system_gbd_data.function_totals[i]
    #     println("  $func_name: $(round(total_val, digits=6))")
    # end

    # println("\nDGBs per atom:")
    # for (atom_idx, atom_sphere_data) in system_gbd_data.atom_sphere_data
    #     if atom_idx <= length(nuclear_cps)
    #         cp = nuclear_cps[atom_idx]
    #         atom_dgbs = [
    #             dgb for dgb in system_gbd_data.differential_gradient_bundles if
    #             dgb.cp_index == atom_idx
    #         ]
    #         if hasfield(typeof(cp), :i_atom) &&
    #             cp.i_atom > 0 &&
    #             cp.i_atom <= length(sys.atoms)
    #             atom_symbol = sys.atoms[cp.i_atom].data.symbol
    #             println("  Atom $atom_idx ($atom_symbol): $(length(atom_dgbs)) DGBs")
    #         else
    #             println("  Atom $atom_idx: $(length(atom_dgbs)) DGBs")
    #         end
    #     end
    # end

    return true
end

function mo_batch()
    root_dir = "/Users/haiiro/scratch/BAND_cleavage/tape41s"

    check_strs = ["000"]

    # walk through all subdirectories in root_dir and for each .t41 file, run mo_test with its full path
    out_data = []
    for (dirpath, dirnames, filenames) in walkdir(root_dir)
        for filename in filenames
            if endswith(filename, ".t41") &&
                (isempty(check_strs) || any(contains(filename, s) for s in check_strs))
                full_path = joinpath(dirpath, filename)
                @info "Processing file: $full_path"
                push!(out_data, test_comprehensive_band_system(full_path))
            end
        end
    end

    return out_data
end
