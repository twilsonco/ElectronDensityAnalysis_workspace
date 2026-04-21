#!/usr/bin/env julia --threads=12

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter: write_zone, open_plt_file, close_plt_file
using LinearAlgebra
using Interpolations: Cubic, Quadratic, Linear
using BenchmarkTools
using PlotlyJS

# Display threading status
@info "Julia threading status" nthreads=Threads.nthreads() physical_cores=Sys.CPU_THREADS÷2

# --- Common Prefix Functions ---
function common_prefix(strings::Vector{String})::String
    if isempty(strings)
        return ""
    end
    # If any string is empty, the common prefix is empty.
    # This must be checked *before* `collect` because `collect("")` is `Char[]`
    # and `minimum(length, ...)` on vectors including an empty one would be 0.
    if any(isempty, strings)
        return ""
    end

    # Convert strings to Vector{Char} to allow 1-based character indexing
    vec_strings = [collect(s) for s in strings]

    # min_len_chars is the minimum number of characters
    min_len_chars = minimum(length, vec_strings)
    # This check is mostly for robustness if the `any(isempty, strings)` was somehow bypassed
    # or if a string became empty after some transformation (not the case here).
    if min_len_chars == 0
        return ""
    end

    prefix_result_chars = Char[] # To build the prefix

    for i in 1:min_len_chars # i is a 1-based character index
        current_char = vec_strings[1][i] # Get i-th char from the first string (now a Vector{Char})

        # Check if this char is the same in all other strings (also Vector{Char}) at this character position
        all_match = true
        for k in 2:length(vec_strings) # Start from the second string
            if vec_strings[k][i] != current_char
                all_match = false
                break
            end
        end
        # Or more compactly:
        # if all(vs[i] == current_char for vs in vec_strings) # Checks all, including the first again

        if all_match # If current_char from first string matches all others at char position i
            push!(prefix_result_chars, current_char)
        else
            break # Mismatch found, stop
        end
    end

    return String(prefix_result_chars)
end

# --- Common Suffix Function ---
function common_suffix(strings::Vector{String})::String
    if isempty(strings) || any(isempty, strings)
        return ""
    end

    reversed_strings = reverse.(strings) # Broadcasting reverse function
    reversed_suffix = common_prefix(reversed_strings) # Using the iterative prefix finder
    return reverse(reversed_suffix)
end

function remove_common_prefix(strings::Vector{String})
    prefix = common_prefix(strings)
    return [s[(length(prefix) + 1):end] for s in strings]
end
function remove_common_suffix(strings::Vector{String})
    suffix = common_suffix(strings)
    return [s[1:(end - length(suffix) - 2)] for s in strings]
end

function plot_sphere_data(gba_info; var_ind=0)
    theta = gba_info["gp_coords"][1, :]
    phi = gba_info["gp_coords"][2, :]
    values = gba_info["gp_int_vals"][var_ind > 0 ? var_ind : end, :]
    var_name = gba_info["var_names"][var_ind > 0 ? var_ind : end]

    # values = gba_info["point_areas"]
    # var_name = "Point Areas"

    @show extrema(theta)
    @show extrema(phi)

    cθ = cos.(phi)
    sθ = sin.(phi)
    cφ = cos.(theta)
    sφ = sin.(theta)

    x = cθ .* sφ
    y = sθ .* sφ
    z = ones(length(theta)) .* cφ

    # f = ones(length(theta)) * df[!,fnames[2]]'
    trace = scatter3d(;
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=attr(;
            size=20, color=values, colorscale="Viridis", colorbar=attr(; title=var_name)
        ),
    )

    layout = Layout(; scene=attr(; xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    return plot(trace, layout)
end

function _find_cps_wrapper(s, sp, dup_check_dist)
    return find_critical_points(s; spacing=sp, dup_cp_check_dist=dup_check_dist)
end

function cp450_test(file_path=nothing)
    path = if file_path === nothing
        "/Users/haiiro/scratch/Cys_propane_NEB_NF_im012_densf_full.t41"
    else
        file_path
    end
    filename_without_ext = basename(path)
    filename_without_ext = splitext(filename_without_ext)[1]

    checkpoint_dir = "/Users/haiiro/scratch/gbd_checkpoints/CP450/$filename_without_ext"

    # sys = load_data_scm_adf_unrestricted_t41(path)
    # return sys

    load_sys_args = (path, load_data_scm_adf_unrestricted_t41)

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys_all = handle_checkpoint(sys_checkpoint_file, with_zip_file, load_sys_args)
    # rhoA, rhoB, rho_total

    sys_sub = [sys_all[2]]
    # return sys_sub

    out_data = []

    original_checkpoint_dir = checkpoint_dir

    for sys in sys_all
        checkpoint_dir = joinpath(original_checkpoint_dir, sys.name)

        @info "Processing system: $(sys.name)" checkpoint_dir

        plt_filename = joinpath(
            dirname(path), "$(filename_without_ext)_$(sys.name)_gbd.plt"
        )

        # If PLT file already exists, continue.
        # if isfile(plt_filename)
        #     @info "PLT file already exists, skipping: $plt_filename"
        #     continue
        # end

        cp_checkpoint_path = joinpath(checkpoint_dir, "cps.jls")

        cp_find_args = (sys, 0.35, 0.1)
        cps = handle_checkpoint(cp_checkpoint_path, _find_cps_wrapper, cp_find_args)

        # Print nuclear CPs with their indices and atom type
        # for (i, cp) in enumerate([c for c in cps if c.type == nuclear_cp])
        #     @info "CP $i: $(cp.type) $(sys.atoms[cp.i_atom].data.symbol)"
        # end

        # # print each atom and its symbol
        # for (i, atom) in enumerate(sys.atoms)
        #     @info "Atom $i: $(atom.data.symbol)"
        # end

        # # print nuclear critical points with their i_atom values
        # for (i, cp) in enumerate([c for c in cps if c.type == nuclear_cp])
        #     @info "Nuclear CP $i: $(cp.i_atom)"
        # end

        # return sys, cps

        # plt = plot_system(
        #     sys;
        #     grid_num_pts=10,
        #     cps=cps,
        #     iso_min=199,
        #     iso_max=200,
        # )

        # display(plt)

        # return sys, cps, plt

        # plt = plot_system(
        #     sys;
        #     grid_num_pts=100,
        # )

        # display(plt)
        # return sys, plt

        @info "Starting gradient bundle decomposition (multithreaded phase)..."
        system_gbd_data = gradient_bundle_decomposition(
            sys;
            cps=cps,
            num_gbs=5000,
            interp_type=Linear,
            func_cutoff=1e-3,
            curvature_threshold=0.95 * 0.5π,
            atom_bounding_box_shrink_factor=0.7,
            ncp_inds=[1],
            seed_sphere_radius_factor=0.1,
            checkpoint_dir=checkpoint_dir,
            system_dependency_files=[sys_checkpoint_file],
            # force_recompute=true,
        )
        @info "Completed gradient bundle decomposition"

        @info "Starting condensed basin finding (iterative phase)..."
        system_gbd_data = find_gradient_bundle_condensed_basins(
            system_gbd_data;
            checkpoint_dir=checkpoint_dir,
            max_iterations=100,
            convergence_count=2,
            # force_recompute=true,
        )
        @info "Completed condensed basin finding"

        plots = []

        # push!(plots, plot_system(
        #     system_gbd_data.system;
        #     grid_num_pts=10,
        #     cps=system_gbd_data.critical_points,
        #     system_gbd_data=system_gbd_data,
        #     plot_condensed_basin_type=condensed_maximum_basin,
        #     # gps=boundary_gps,
        #     iso_min=100,
        #     iso_max=200,
        # ))

        # push!(plots, plot_system(
        #     system_gbd_data.system;
        #     grid_num_pts=10,
        #     cps=system_gbd_data.critical_points,
        #     system_gbd_data=system_gbd_data,
        #     plot_condensed_basin_type=condensed_maximum_basin,
        #     # gps=boundary_gps,
        #     iso_min=100,
        #     iso_max=200,
        #     gba_func_num=2,
        # ))

        # system_gbd_data = refine_condensed_basins(
        #     system_gbd_data;
        #     # Consolidation parameters
        #     do_consolidation=true,
        #     threshold_fraction=0.7,
        #     improper_threshold=0.05,
        #     # Sliver removal parameters
        #     do_sliver_removal=true,
        #     # Boundary flag update
        #     update_boundaries=true,
        #     checkpoint_dir=checkpoint_dir,
        # )

        # Generate boundary gradient paths for all maximum basins
        # boundary_gps_data = generate_system_condensed_boundary_gradient_paths(
        #     system_gbd_data;
        #     gp_func_cutoff=1e-3,
        #     gp_direction=both_dir,
        # )

        # # Extract boundary gradient paths only from atoms 1, 9, and 10
        # target_atoms = [1, 4, 9, 10]
        # boundary_gps = GradientPath[]
        # for ((cp_index, func_index, region_type), basin_data) in boundary_gps_data
        #     if region_type == condensed_maximum_basin && cp_index in target_atoms
        #         for (basin_focus, (edge_gp_map, perimeter_edges)) in basin_data
        #             for gp in values(edge_gp_map)
        #                 # Adaptively resample the gradient path to 20 points
        #                 resampled_gp = resample_adaptive(gp, 20)
        #                 push!(boundary_gps, resampled_gp)
        #             end
        #         end
        #     end
        # end

        # @info "Boundary gradient paths count: $(length(boundary_gps))"

        push!(
            plots,
            plot_system(
                system_gbd_data.system;
                grid_num_pts=10,
                cps=system_gbd_data.critical_points,
                system_gbd_data=system_gbd_data,
                plot_condensed_basin_type=condensed_maximum_basin,
                # gps=boundary_gps,
                iso_min=199,
                iso_max=200,
                # atom_inds=[53],
            ),
        )

        # push!(plots, plot_system(
        #     system_gbd_data.system;
        #     grid_num_pts=10,
        #     cps=system_gbd_data.critical_points,
        #     system_gbd_data=system_gbd_data,
        #     plot_condensed_basin_type=condensed_maximum_basin,
        #     # gps=boundary_gps,
        #     iso_min=100,
        #     iso_max=200,
        #     gba_func_num=2,
        # ))

        print_atom_function_totals(system_gbd_data)

        # Write Tecplot PLT file for this system (adjacent to original file)
        @info "Writing Tecplot PLT file: $plt_filename"

        variables = ["X", "Y", "Z"]
        append!(variables, system_gbd_data.function_names)

        try
            open_plt_file(plt_filename, "GBD Analysis: $(sys.name)", join(variables, ", "))

            write_zone(
                "$(sys.name) GBD",
                system_gbd_data;
                system=sys,
                variables=variables,
                cell_centered_spheres=false,
                print_dgbs=true,
                print_condensed_basin_boundary_gps=true,
                print_condensed_basin_spheres=true,
            )

            close_plt_file()
            @info "Successfully wrote PLT file: $plt_filename"

            file_size_mb = round(filesize(plt_filename) / (1024 * 1024); digits=2)
            @info "File size: $file_size_mb MB"
        catch e
            @error "Failed to write PLT file: $plt_filename" exception=(
                e, catch_backtrace()
            )
            close_plt_file()
        end

        push!(out_data, [system_gbd_data, plots])
    end

    return out_data

    # print_critical_points_info(system_gbd_data.system, system_gbd_data.critical_points)
    # print_atom_function_totals(system_gbd_data)
    # print_atom_basin_totals(system_gbd_data; basin_type=condensed_maximum_basin)

    # return system_gbd_data, plt
end

function cp450_batch()
    root_dir = "/Users/twilson/scratch/CP_crossings/Cys/for_bondalyzer"

    # walk through all subdirectories in root_dir and for each .t41 file, run cp450_test with its full path
    out_data = []
    for (dirpath, dirnames, filenames) in walkdir(root_dir)
        for filename in filenames
            if endswith(filename, ".t41")
                full_path = joinpath(dirpath, filename)
                @info "Processing file: $full_path"
                push!(out_data, cp450_test(full_path))
            end
        end
    end

    return out_data
end

function cp450_show(data)
    for sys in data
        # display(sys[1][2][4])
        display(sys[2][2][1])
        # display(sys[3][2][4])
    end
end

function ethanol_test()
    checkpoint_dir = "/Volumes/HaiiroStudio_Ext_6T/gbd_checkpoints/Ethanol"
    # sys = with_zip_file("test_data/ethanol_adf.t41", load_data_scm_adf_t41)
    load_sys_args = (
        "test_data/ethanol_adf.t41", (f) -> with_zip_file(f, load_data_scm_adf_t41)
    )

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys = handle_checkpoint(sys_checkpoint_file, with_zip_file, load_sys_args)

    # return sys

    system_gbd_data = gradient_bundle_decomposition(
        sys;
        num_gbs=5000,
        interp_type=Linear,
        func_cutoff=1e-3,
        curvature_threshold=0.8 * 0.5π,
        atom_bounding_box_shrink_factor=0.8,
        checkpoint_dir=checkpoint_dir,
        system_dependency_files=[sys_checkpoint_file],
        # force_recompute=true,
    )

    system_gbd_data = find_gradient_bundle_condensed_basins(
        system_gbd_data;
        checkpoint_dir=checkpoint_dir,
        max_iterations=100,
        convergence_count=6,
        # force_recompute=true,
    )

    plt = plot_system(
        system_gbd_data.system;
        grid_num_pts=30,
        cps=system_gbd_data.critical_points,
        system_gbd_data=system_gbd_data,
        plot_condensed_basin_type=condensed_maximum_basin,
    )
    display(plt)

    print_atom_function_totals(system_gbd_data)

    print_atom_basin_totals(system_gbd_data; basin_type=condensed_maximum_basin)
    print_atom_basin_totals(system_gbd_data; basin_type=condensed_minimum_basin)

    return system_gbd_data#, plt
end

function pu_test()
    # checkpoint_dir = "/Volumes/HaiiroStudio_Ext_6T/gbd_checkpoints/Al"
    # load_sys_args = ("/Users/haiiro/NoSync/Al_005.t41", load_data_scm_band_t41)

    in_file = "CHGCAR_3Q"
    checkpoint_dir = "/Volumes/HaiiroStudio_Ext_6T/gbd_checkpoints/Pu_$(in_file)"
    load_sys_args = (
        "/Users/haiiro/NoSync/2025_travis_pu/$(in_file)", load_data_vasp_chgcar
    )

    # sys = load_data_vasp_chgcar(load_sys_args[1])
    # return sys

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys_original = handle_checkpoint(sys_checkpoint_file, with_zip_file, load_sys_args)

    return sys_original

    # plt = plot_system(
    #     sys[1];
    #     grid_num_pts=30,
    #     # cps=cps,
    #     iso_min=1e-2,
    #     iso_max=50,
    # )

    # display(plt)

    # return sys

    sys = [translate_system_periodic(s, [0.5, 0.0, 0.0]) for s in sys_original]
    sys = [
        expand_system_periodic(s; factor=1.2, is_periodic=[true, true, true]) for s in sys
    ]

    # gp = create_gradient_path([0.1, 0.1, 0.1], sys[1], backward_dir; max_steps = 10000)

    # return sys[1], gp

    # plt = plot_system(
    #     sys[1];
    #     grid_num_pts=100,
    #     # cps=cps,
    #     iso_min=1e-3,
    #     iso_max=10,
    # )

    # display(plt)

    # return sys

    # sys = with_zip_file("$DATA_DIR/benzene_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("test_data/ethanol_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("test_data/adamantane_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("test_data/h2_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("/Users/haiiro/NoSync/Al_005.t41", load_data_scm_band_t41)
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/BondalyzerPrototype.jl/data/sims/Al_005.cub",
    #     load_data_cube;
    #     remove_unzipped_file=false,
    # )
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/BondalyzerPrototype.jl/data/sims/adamantane-fine.cub",
    #     load_data_cube,
    # )
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/BondalyzerPrototype.jl/data/sims/He-fine.cub", f->load_data_cube(f); remove_unzipped_file=false
    # )
    # sys = with_zip_file("test_data/buckyball_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("test_data/Ne2_adf.t41", (f) -> load_data_scm_adf_t41(f; T=Float32))
    # sys, f3d, f3d_xyz, grad3d, hess3d, func!, grad!, hess! = create_trial_3D_system_interpolated()
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/CHARGE-3D-0.05.flapw.charge",
    #     (f) -> load_data_flapw_charge(f; T=Float64),
    # )

    # Get list of system names, with the common prefix removed
    sys_names = map(s -> s.name, sys)
    sys_names = remove_common_prefix(sys_names)
    sys_names = remove_common_suffix(sys_names)

    original_checkpoint_dir = checkpoint_dir

    out_data = []

    for i in eachindex(sys)
        s = sys[i]

        checkpoint_dir = joinpath(original_checkpoint_dir, sys_names[i])

        cp_checkpoint_path = joinpath(checkpoint_dir, "cps.jls")

        cp_find_args = (s, 0.35, 0.1)
        cps = handle_checkpoint(cp_checkpoint_path, _find_cps_wrapper, cp_find_args)

        plt = plot_system(s; grid_num_pts=50, cps=cps, iso_min=1e-3, iso_max=10)

        display(plt)

        continue

        system_gbd_data = gradient_bundle_decomposition(
            s;
            cps=cps,
            num_gbs=20000,
            interp_type=Linear,
            func_cutoff=1e-5,
            curvature_threshold=0.98 * 0.5π,
            seed_sphere_radius_factor=0.1,
            atom_bounding_box_shrink_factor=0.2,
            checkpoint_dir=checkpoint_dir,
            system_dependency_files=[sys_checkpoint_file],
            force_recompute=true,
        )

        # return system_gbd_data

        system_gbd_data = find_gradient_bundle_condensed_basins(
            system_gbd_data;
            checkpoint_dir=checkpoint_dir,
            max_iterations=100,
            convergence_count=6,
            # force_recompute=true,
        )

        # system_gbd_data = refine_condensed_basins(
        #     system_gbd_data;
        #     # Consolidation parameters
        #     do_consolidation=true,
        #     threshold_fraction=0.7,
        #     improper_threshold=0.05,
        #     # Sliver removal parameters
        #     do_sliver_removal=false,
        #     # Boundary flag update
        #     update_boundaries=true,
        # )

        # condensed_basin_boundary_gps_data = generate_system_condensed_boundary_gradient_paths(system_gbd_data)

        plt = plot_system(
            system_gbd_data.system;
            grid_num_pts=10,
            cps=system_gbd_data.critical_points,
            iso_min=199,
            iso_max=200,
            system_gbd_data=system_gbd_data,
            plot_condensed_basin_type=condensed_minimum_basin,
            # cell_centered_spheres=true,
            # plot_all_condensed_basin_gps=true, # Plot all condensed basin GPs
            # condensed_basin_boundary_gps=condensed_basin_boundary_gps_data,
        )
        display(plt)

        push!(out_data, [system_gbd_data, plt])
        break
    end

    # Loop over out_data to print atomic basin and condensed basin totals
    for (system_gbd_data, plt) in out_data
        print_atom_function_totals(system_gbd_data)
        print_atom_basin_totals(system_gbd_data; basin_type=condensed_minimum_basin)
    end

    return out_data
end

function mo_test(file_path=nothing)
    path = if file_path === nothing
        "/Users/twilson/scratch/Cleavage/Restarts/Mo4_140_restart_fine.results/Mo4_140_restart_fine.t41"
    else
        file_path
    end
    filename_without_ext = splitext(basename(path))[1]

    plt_filename = joinpath(dirname(path), "$(filename_without_ext)_gbd.plt")
    # Only write plt file if it doesn't already exist
    # if isfile(plt_filename)
    #     @info "PLT file already exists, skipping: $plt_filename"
    #     return nothing
    # end

    checkpoint_dir = "/Users/twilson/scratch/gbd_checkpoints/Mo_cleavage/$filename_without_ext"

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys_original = load_data_scm_band_t41(path)[1]

    # sys_tau = load_data_scm_band_t41(path; load_var="tau")[1]
    # sys_elf = load_data_scm_band_t41(path; load_var="elf[rho]")[1]

    sys = sys_original

    # @info "Atom count before mirroring: $(length(sys_original.atoms))"

    # sys = mirror_system_periodic(
    #     sys_original; mirror_dims=[true, true, false], origin_at_grid_point=false
    # )

    # @info "Atom count after mirroring: $(length(sys.atoms))"

    # Print all atom info
    for (i, atom) in enumerate(sys.atoms)
        @info "Atom $i: $(atom.data.symbol) Position: $(atom.r)"
    end

    # plt_filename = joinpath(dirname(path), "$(filename_without_ext).plt")

    # variables = ["X", "Y", "Z", "Electron Density"]

    # open_plt_file(plt_filename, "GBD Analysis: Mo_100", join(variables, ", "))

    # write_zone(
    #     "Mo GBD",
    #     sys_original;
    #     system=sys_original,
    #     variables=variables,
    # )

    # close_plt_file()

    # return

    @info "Processing Mo system: $(sys.name)" checkpoint_dir

    # Extend system in X and Y directions by factor of 1.5
    # sys = expand_system_periodic(sys; factor=1.5, is_periodic=[true, true, false])

    # @info "Extended system dimensions" grid_size=sys.grid.n_pts

    # Find critical points
    @info "Finding critical points..."
    cp_checkpoint_path = joinpath(checkpoint_dir, "cps.jls")
    cp_find_args = (sys, 0.25, 0.2)
    cps = handle_checkpoint(cp_checkpoint_path, _find_cps_wrapper, cp_find_args)
    @info "Found $(length(cps)) critical points"

    # Perform gradient bundle decomposition
    @info "Starting gradient bundle decomposition (multithreaded phase)..."
    system_gbd_data = gradient_bundle_decomposition(
        sys;
        cps=cps,
        num_gbs=5000,
        interp_type=Linear,
        func_cutoff=1e-3,
        curvature_threshold=0.95 * 0.5π,
        atom_bounding_box_shrink_factor=0.7,
        seed_sphere_radius_factor=0.3,
        checkpoint_dir=checkpoint_dir,
        system_dependency_files=[sys_checkpoint_file],
        # f_list_in=[("tau", sys_tau.func), ("elf", sys_elf.func)],
        force_recompute=false,
    )
    @info "Completed gradient bundle decomposition"

    # Find condensed basins
    @info "Starting condensed basin finding (iterative phase)..."
    system_gbd_data = find_gradient_bundle_condensed_basins(
        system_gbd_data;
        checkpoint_dir=checkpoint_dir,
        max_iterations=30,
        convergence_count=2,
        force_recompute=false,
    )

    system_gbd_data = refine_condensed_basins(
        system_gbd_data; checkpoint_dir=checkpoint_dir, force_recompute=false
    )
    @info "Completed condensed basin finding"

    plt_filename = joinpath(dirname(path), "$(filename_without_ext)_gbd.plt")

    # Write Tecplot PLT file
    @info "Writing Tecplot PLT file: $plt_filename"

    variables = ["X", "Y", "Z"]

    append!(variables, system_gbd_data.function_names)
    push!(variables, "α")

    output_path_base = joinpath(dirname(path), "$(filename_without_ext)")

    print_system_function_total(sys; output_path_base=output_path_base)
    print_atom_function_totals(system_gbd_data; output_path_base=output_path_base)
    print_atom_basin_totals(
        system_gbd_data;
        basin_type=condensed_maximum_basin,
        output_path_base=output_path_base,
    )
    print_atom_basin_totals(
        system_gbd_data;
        basin_type=condensed_minimum_basin,
        output_path_base=output_path_base,
    )

    # Only write plt file if it doesn't already exist
    if isfile(plt_filename)
        @info "PLT file already exists, skipping: $plt_filename"
        return system_gbd_data
    end

    try
        open_plt_file(plt_filename, "GBD Analysis: Mo_100", join(variables, ", "))

        write_zone(
            "Mo GBD",
            system_gbd_data;
            system=sys,
            variables=variables,
            cell_centered_spheres=false,
            print_condensed_basin_boundary_gps=true,
            print_condensed_basin_spheres=true,
            print_dgbs=false,
        )

        close_plt_file()
        @info "Successfully wrote PLT file: $plt_filename"

        file_size_mb = round(filesize(plt_filename) / (1024 * 1024); digits=2)
        @info "File size: $file_size_mb MB"
    catch e
        @error "Failed to write PLT file: $plt_filename" exception=(e, catch_backtrace())
        close_plt_file()
    end

    return system_gbd_data
end

function mo_batch()
    root_dir = "/Users/twilson/scratch/Cleavage/Restarts"

    check_strs = ["200"]

    # walk through all subdirectories in root_dir and for each .t41 file, run mo_test with its full path
    out_data = []
    for (dirpath, dirnames, filenames) in walkdir(root_dir)
        for filename in filenames
            if endswith(filename, ".t41") &&
                (isempty(check_strs) || any(contains(filename, s) for s in check_strs))
                full_path = joinpath(dirpath, filename)
                @info "Processing file: $full_path"
                push!(out_data, mo_test(full_path))
            end
        end
    end

    return out_data
end

function plot_gbd_spheres(sgbd_data, system_number::Int)
    sgbd = sgbd_data[system_number]
    plot_system(
        sgbd.system;
        grid_num_pts=10,
        cps=sgbd.critical_points,
        iso_min=199,
        iso_max=200,
        system_gbd_data=sgbd,
        plot_condensed_basin_type=condensed_maximum_basin,
    )
end


function adf_test(file_path)
    filename_without_ext = basename(file_path)
    filename_without_ext = splitext(filename_without_ext)[1]

    # Checkpointdir will be "checkpoints" dir adjacent to the input file, with a subdir for the system name (with common prefix/suffix removed if multiple systems)
    checkpoint_dir = joinpath(dirname(file_path), "checkpoints", filename_without_ext)

    # sys = load_data_scm_adf_unrestricted_t41(path)
    # return sys

    load_sys_args = (file_path, load_data_scm_adf_t41)

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys_all = handle_checkpoint(sys_checkpoint_file, with_zip_file, load_sys_args)
    sys = sys_all[1]

    # load kinetic energy density system if available
    sys_kinetic = load_data_scm_adf_t41(file_path; load_var="Kinetic Energy Density")[1]
    f_list = [("Kinetic Energy Density", sys_kinetic.func)]

    cp_checkpoint_path = joinpath(checkpoint_dir, "cps.jls")

    cp_find_args = (sys, 0.35, 0.1)
    cps = handle_checkpoint(cp_checkpoint_path, _find_cps_wrapper, cp_find_args)

    # Print nuclear CPs with their indices and atom type
    for (i, cp) in enumerate([c for c in cps if c.type == nuclear_cp])
        @info "CP $i: $(cp.type) $(sys.atoms[cp.i_atom].data.symbol)"
    end

    # return sys

    system_gbd_data = gradient_bundle_decomposition(
        sys;
        num_gbs=5000,
        interp_type=Linear,
        seed_sphere_radius_factor=0.25,
        func_cutoff=1e-3,
        curvature_threshold=0.8 * 0.5π,
        ncp_inds=[1],
        f_list_in=f_list,
        # atom_bounding_box_shrink_factor=0.8,
        checkpoint_dir=checkpoint_dir,
        system_dependency_files=[sys_checkpoint_file],
        force_recompute=true,
    )

    system_gbd_data = find_gradient_bundle_condensed_basins(
        system_gbd_data;
        checkpoint_dir=checkpoint_dir,
        max_iterations=100,
        convergence_count=4,
        force_recompute=true,
    )

    print_atom_function_totals(system_gbd_data)

    print_atom_basin_totals(system_gbd_data; basin_type=condensed_maximum_basin)
    print_atom_basin_totals(system_gbd_data; basin_type=condensed_minimum_basin)

    variables = ["X", "Y", "Z"]
    append!(variables, system_gbd_data.function_names)

    print("Variables to be written to PLT file: ", variables)

    # return

    plt_filename = joinpath(
        dirname(file_path), "$(filename_without_ext)_$(sys.name)_gbd.plt"
    )

    try
        open_plt_file(plt_filename, "GBD Analysis: $(sys.name)", join(variables, ", "))

        write_zone(
            "$(sys.name) GBD",
            system_gbd_data;
            system=sys,
            variables=variables,
            cell_centered_spheres=false,
            print_dgbs=false,
            print_condensed_basin_boundary_gps=false,
            print_condensed_basin_spheres=true,
        )

        close_plt_file()
        @info "Successfully wrote PLT file: $plt_filename"

        file_size_mb = round(filesize(plt_filename) / (1024 * 1024); digits=2)
        @info "File size: $file_size_mb MB"
    catch e
        @error "Failed to write PLT file: $plt_filename" exception=(
            e, catch_backtrace()
        )
        close_plt_file()
    end

    return system_gbd_data#, plt
end

function adf_test_batch()
    root_dir = "/Users/haiiro/scratch/BH3-NH3_PES-scan.results/SPs"

    # walk through all subdirectories in root_dir and for each .t41 file, run adf_test with its full path
    out_data = []
    for (dirpath, dirnames, filenames) in walkdir(root_dir)
        for filename in filenames
            if endswith(filename, ".t41")
                full_path = joinpath(dirpath, filename)
                @info "Processing file: $full_path"
                push!(out_data, adf_test(full_path))
            end
        end
    end

    return out_data
end