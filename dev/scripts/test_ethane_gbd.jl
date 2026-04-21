using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Cubic, Quadratic, Linear
using BenchmarkTools
using PlotlyJS
include("../test/test_helpers.jl")

function ethane_test()
    checkpoint_dir = "/Volumes/HaiiroStudio_Ext_6T/gbd_checkpoints/Ethane"
    # sys = with_zip_file("test_data/ethanol_adf.t41", load_data_scm_adf_t41)
    load_sys_args = (
        "test_data/ethane_adf.t41", (f) -> with_zip_file(f, load_data_scm_adf_t41)
    )

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys = handle_checkpoint(sys_checkpoint_file, with_zip_file, load_sys_args)[1]

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

    system_gbd_data = refine_condensed_basins(system_gbd_data)

    condensed_basin_boundary_gps_data = generate_system_condensed_boundary_gradient_paths(
        system_gbd_data
    )

    plt = plot_system(
        system_gbd_data.system;
        grid_num_pts=30,
        cps=system_gbd_data.critical_points,
        system_gbd_data=system_gbd_data,
        plot_condensed_basin_type=condensed_maximum_basin,
        condensed_basin_boundary_gps=condensed_basin_boundary_gps_data,
    )
    display(plt)

    print_atom_function_totals(system_gbd_data)

    print_atom_basin_totals(system_gbd_data; basin_type=condensed_maximum_basin)
    print_atom_basin_totals(system_gbd_data; basin_type=condensed_minimum_basin)

    return system_gbd_data#, plt
end

ethane_test();
