using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Cubic, Quadratic, Linear
using BenchmarkTools
using PlotlyJS
using PyFormattedStrings

using PlotlyJS: mgrid, isosurface, attr, Layout, plot, scatter, savefig, SyncPlot
using SplitApplyCombine: invert

using Serialization

include("../test/test_helpers.jl")

function main(N=20000)
    # sys = with_zip_file("test_data/benzene_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("/Users/haiiro/NoSync/Al_005.t41", load_data_scm_band_t41)
    # sys = with_zip_file("/Users/haiiro/NoSync/Cu_GB_fit_coarse.t41", (f) -> load_data_scm_band_t41(f; load_var="rho(fit)"))

    sys = load_data_scm_band_t41("/Users/haiiro/NoSync/Cu_GB_coarse.t41"; load_var="rho")[1]
    # sys_tau = load_data_scm_band_t41(
    #     "/Users/haiiro/NoSync/Cu_GB_coarse.t41"; load_var="tau"
    # )

    # sys = with_zip_file("/Users/haiiro/NoSync/cu43-rotated-fine.t41", load_data_scm_adf_t41)

    # sys = with_zip_file("test_data/ethanol_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("test_data/adamantane_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("test_data/h2_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/BondalyzerPrototype.jl/data/sims/Al_005.cub",
    #     load_data_cube;
    #     remove_unzipped_file=false,
    # )
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/BondalyzerPrototype.jl/data/sims/adamantane-fine.cub",
    #     load_data_cube;
    #     remove_unzipped_file=false,
    # )
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/BondalyzerPrototype.jl/data/sims/He-fine.cub", f->load_data_cube(f); remove_unzipped_file=false
    # )
    # sys = with_zip_file("test_data/buckyball_adf.t41", load_data_scm_adf_t41)
    # sys = with_zip_file("test_data/Ne2_adf.t41", (f) -> load_data_scm_adf_t41(f; T=Float32))
    # sys, f3d, f3d_xyz, grad3d, hess3d, func!, grad!, hess! = create_trial_3D_system_interpolated()
    # sys = with_zip_file(
    #     "/Users/haiiro/NoSync/CHARGE-3D-0.05.flapw.charge",
    #     (f) -> load_data_flapw_charge(
    #         f; T=Float64, z_spacing=0.025, is_periodic=[false, false, false]
    #     ),
    # )

    ###################
    # Redo the grid to account for uneven spacing in the Z direction
    # new_data = sys.data[:, :, begin:2:end]
    # new_size = size(new_data)
    # lattice = copy(sys.grid.lattice)
    # lattice[:, 3] *= 2
    # new_gs = GridSpec(sys.grid.origin, lattice, Vector{Int}(collect(new_size)))
    # @info "New grid" new_gs
    # func_xyz_f, func, grad, hess, func!, grad!, hess! = create_interpolated_fields(
    #     new_data, new_gs; interp_type=Quadratic
    # )
    # sys = System(
    #     sys.name,
    #     sys.source,
    #     sys.atoms,
    #     new_gs,
    #     new_data,
    #     [false, false, false],
    #     func_xyz_f,
    #     func,
    #     grad,
    #     hess,
    #     func!,
    #     grad!,
    #     hess!,
    # )
    ###################

    ###################
    # Expand periodic system so that we can work in a larger central space
    sys = expand_system_periodic(sys; factor=2.5, is_periodic=[true, true, false])
    # sys_tau = expand_system_periodic(sys_tau; factor=2.5, is_periodic=[true, true, false])

    # Save the system to a file using Serialization
    open("temp_data/sys.jls", "w") do file
        serialize(file, sys)
    end
    # open("temp_data/sys_tau.jls", "w") do file
    #     serialize(file, sys_tau)
    # end
    ###################

    # return sys

    # return plot_system(sys)

    cps = find_critical_points(sys; spacing=0.35, bounding_box_shrink_factor=0.8)
    open("temp_data/cps.jls", "w") do file
        serialize(file, cps)
    end

    plt = plot_system(sys; cps=cps)

    return plt

    # bond_paths = create_bond_paths(sys, cps)
    # # ring_paths = create_ring_paths(sys, cps)
    # # cage_paths = create_cage_paths(sys, cps)

    # plt = plot_system(
    #     sys;
    #     # cps=cps,
    #     # system_gbd_data=system_gbd_data,
    #     # condensed_basins=max_basins[1],
    #     # gps=out_of_bounds_gps,
    #     # bond_paths=bond_paths,
    #     # ring_paths=ring_paths,
    #     # cage_paths=cage_paths,
    #     iso_min=1e-3,
    #     # cp_type_blacklist=[ring_cp,cage_cp]
    # )

    # return sys, plt

    n = N
    system_gbd_data = gradient_bundle_decomposition(
        sys;
        num_gbs=n,
        num_non_nuclear_seed_points=0,#ceil(Int, 2sqrt(n)),
        func_cutoff=1e-3,
        path_saddle_min_dist=0.02,
        gp_filling_max_num_iters=1,
        type="whole_system",
        cp_spacing=0.35,
        cp_bounding_box_shrink_factor=0.6,
        atom_bounding_box_shrink_factor=0.2,
        f_sys_list=[sys_tau],
    )

    # Save the system_gbd_data to a file using Serialization
    open("temp_data/gbd_data.jls", "w") do file
        serialize(file, system_gbd_data)
    end

    # return nothing

    # return system_gbd_data

    # compute min and max basins using find_gradient_bundle_condensed_basins(sys_gbd_data::SystemGBDDataOld)
    max_basins, min_basins = find_gradient_bundle_condensed_basins(
        system_gbd_data; max_iterations=100
    )

    # max_basins, min_basins = refine_condensed_basins(
    #     system_gbd_data, max_basins, min_basins
    # )

    # Get all gradient paths that terminated with an out_of_bounds status
    out_of_bounds_gps = []
    for basin in values(system_gbd_data.basin_gbd_data)
        for gp in basin.gradient_paths
            gp.term_status == out_of_bounds && push!(out_of_bounds_gps, gp)
        end
    end

    bond_paths = create_bond_paths(sys, system_gbd_data.critical_points)
    # ring_paths = create_ring_paths(sys, system_gbd_data.critical_points)
    # cage_paths = create_cage_paths(sys, system_gbd_data.critical_points)

    # plt_cb = plot_system(
    #     sys;
    #     cps=system_gbd_data.critical_points,
    #     system_gbd_data=system_gbd_data,
    #     condensed_basins=max_basins[1],
    #     gps=out_of_bounds_gps,
    #     bond_paths=bond_paths,
    #     ring_paths=ring_paths,
    #     # cage_paths=cage_paths,
    #     iso_min=1e-3,
    #     cp_type_blacklist=[ring_cp,cage_cp]
    # )
    plt = plot_system(
        sys;
        cps=system_gbd_data.critical_points,
        system_gbd_data=system_gbd_data,
        gba_func_num=2,
        bond_paths=bond_paths,
        # ring_paths=ring_paths,
        # cage_paths=cage_paths,
        gps=out_of_bounds_gps,
        iso_min=10,
        iso_max=100,
        cp_type_blacklist=[bond_cp, ring_cp, cage_cp],
    )
    return system_gbd_data, max_basins, min_basins, plt
end

function gbd_test_restart()
    # Load the system from a file using Serialization
    sys = open(deserialize, "temp_data/sys.jls")
    # sys_tau = open(deserialize, "temp_data/sys_tau.jls")

    cps = open(deserialize, "temp_data/cps.jls")

    system_gbd_data = gradient_bundle_decomposition(
        sys;
        cps=cps,
        num_gbs=5000,
        num_non_nuclear_seed_points=0,
        func_cutoff=1e-3,
        path_saddle_min_dist=0.02,
        gp_filling_max_num_iters=1,
        type="whole_system",
        cp_spacing=0.35,
        # cp_bounding_box_shrink_factor=0.6,
        atom_bounding_box_shrink_factor=0.4,
        # f_sys_list=[sys_tau],
        ncp_min_func_cutoff=1.0,
    )

    # Save the system_gbd_data to a file using Serialization
    open("temp_data/gbd_data.jls", "w") do file
        serialize(file, system_gbd_data)
    end

    plt = plot_system(
        sys; cps=system_gbd_data.critical_points, system_gbd_data=system_gbd_data
    )

    return system_gbd_data, plt
end

function main1()
    system_gbd_data = open(deserialize, "temp_data/gbd_data.jls")

    max_basins, min_basins = find_gradient_bundle_condensed_basins(
        system_gbd_data; max_iterations=100, cp_numbers=[32]
    )

    # max_basins, min_basins = refine_condensed_basins(
    #     system_gbd_data, max_basins, min_basins
    # )

    # Get all gradient paths that terminated with an out_of_bounds status
    out_of_bounds_gps = []
    for basin in values(system_gbd_data.basin_gbd_data)
        for gp in basin.gradient_paths
            gp.term_status == out_of_bounds && push!(out_of_bounds_gps, gp)
        end
    end

    bond_paths = create_bond_paths(system_gbd_data.system, system_gbd_data.critical_points)
    # ring_paths = create_ring_paths(sys, system_gbd_data.critical_points)
    # cage_paths = create_cage_paths(sys, system_gbd_data.critical_points)

    # plt_cb = plot_system(
    #     sys;
    #     cps=system_gbd_data.critical_points,
    #     system_gbd_data=system_gbd_data,
    #     condensed_basins=max_basins[1],
    #     gps=out_of_bounds_gps,
    #     bond_paths=bond_paths,
    #     ring_paths=ring_paths,
    #     # cage_paths=cage_paths,
    #     iso_min=1e-3,
    #     cp_type_blacklist=[ring_cp,cage_cp]
    # )
    plt = plot_system(
        system_gbd_data.system;
        cps=system_gbd_data.critical_points,
        system_gbd_data=system_gbd_data,
        gba_func_num=2,
        bond_paths=bond_paths,
        # ring_paths=ring_paths,
        # cage_paths=cage_paths,
        gps=out_of_bounds_gps,
        iso_min=10,
        iso_max=100,
        # cp_type_blacklist=[bond_cp, ring_cp, cage_cp],
    )
    return system_gbd_data, max_basins, min_basins, plt
end

function find_plot_condensed_basins(
    system_gbd_data; max_iterations=100, convergence_count=6
)
    max_basins, min_basins = find_gradient_bundle_condensed_basins(
        system_gbd_data; max_iterations=max_iterations, convergence_count=convergence_count
    )

    validate_basin_assignments(system_gbd_data, max_basins)
    validate_basin_assignments(system_gbd_data, min_basins)

    # Get all gradient paths that terminated with an out_of_bounds status
    out_of_bounds_gps = []
    for basin in values(system_gbd_data.basin_gbd_data)
        for gp in basin.gradient_paths
            gp.term_status == out_of_bounds && push!(out_of_bounds_gps, gp)
        end
    end

    bond_paths = create_bond_paths(system_gbd_data.system, system_gbd_data.critical_points)
    ring_paths = create_ring_paths(system_gbd_data.system, system_gbd_data.critical_points)
    cage_paths = create_cage_paths(system_gbd_data.system, system_gbd_data.critical_points)

    max_basins, min_basins = refine_condensed_basins(
        system_gbd_data, max_basins, min_basins
    )

    boundaries = generate_system_condensed_boundary_gradient_paths(
        system_gbd_data, [min_basins, max_basins]
    )

    plt_cb = plot_system(
        system_gbd_data.system;
        cps=system_gbd_data.critical_points,
        system_gbd_data=system_gbd_data,
        condensed_basins=max_basins[1],
        gps=out_of_bounds_gps,
        bond_paths=bond_paths,
        ring_paths=ring_paths,
        cage_paths=cage_paths,
        iso_min=1e-3,
    )
    plt = plot_system(
        system_gbd_data.system;
        cps=system_gbd_data.critical_points,
        system_gbd_data=system_gbd_data,
        bond_paths=bond_paths,
        ring_paths=ring_paths,
        cage_paths=cage_paths,
        gps=out_of_bounds_gps,
        iso_min=1e-3,
    )
    boundary_gps = [collect(values(atom[1][1])) for atom in boundaries[2]]
    # flatten the boundary_gps into a single list
    boundary_gps = [
        boundary_gps[i][j] for i in 1:length(boundary_gps) for
        j in 1:length(boundary_gps[i])
    ]
    plt_w_boundary = plot_system(
        system_gbd_data.system;
        cps=system_gbd_data.critical_points,
        system_gbd_data=system_gbd_data,
        condensed_basins=max_basins[1],
        bond_paths=bond_paths,
        ring_paths=ring_paths,
        cage_paths=cage_paths,
        gps=boundary_gps,
        iso_min=1e-3,
    )

    return system_gbd_data, max_basins, min_basins, plt, plt_cb, plt_w_boundary
end

function refine_condensed_basins(system_gbd_data, max_basins, min_basins)
    max_basins = consolidate_improper_basins(system_gbd_data, max_basins)
    min_basins = consolidate_improper_basins(system_gbd_data, min_basins)
    return max_basins, min_basins
end

function print_sys_gbd_info(system_gbd_data)
    for atom_data in values(system_gbd_data.basin_gbd_data)
        println(
            "\nAtom: $(system_gbd_data.system.atoms[atom_data.atom_number].data.symbol)$(atom_data.atom_number)",
        )
        println("Critical point number: $(atom_data.critical_point_number)")
        println("Sphere radius: $(atom_data.sphere_radius)")
        println("Sphere coordinates size: $(size(atom_data.gb_sphere_coordinates))")
        println("Sphere points size: $(size(atom_data.gb_sphere_points))")
        for func in atom_data.gb_condensed_functions
            println("Function: $(func.function_name)")
            println("Basin total: $(func.basin_total)")
            println("Sphere only: $(func.basin_total_sphere_only)")
            println("Number of gradient bundles: $(length(func.gb_values))")
            println("Sum of GB values: $(sum(func.gb_values))")
            println(
                "Sum of area-normalized GB values: $(sum(func.gb_values_area_normalized))"
            )
        end
    end
    println("\nSystem totals:")
    for (fi, func) in enumerate(system_gbd_data.function_names)
        println("  Function: $(func), total: $(system_gbd_data.function_totals[fi])")
    end
end

function print_gbd_basin_info(system_gbd_data, basins)
    # print info for the max basins of the first condensed funciton for the first atom
    println(keys(system_gbd_data.basin_gbd_data))
    first_basin = collect(keys(system_gbd_data.basin_gbd_data))[1]
    println(
        "$(length(basins[1][first_basin])) max basins found in function $(system_gbd_data.function_names[1]) for atom $(system_gbd_data.system.atoms[first_basin].data.symbol)$(first_basin)",
    )

    rho_total = 0.0
    for (basin_node, basin_nodes) in basins[1][first_basin]
        # println("    Basin max node: $(basin_node)")
        # println("    Basin max node value: $(system_gbd_data.basin_gbd_data[first_basin].gb_condensed_functions[1].gb_values[basin_node])")
        # Get rho total for the basin
        basin_rho_total = 0.0
        for node in basin_nodes
            basin_rho_total += system_gbd_data.basin_gbd_data[first_basin].gb_condensed_functions[1].gb_values[node]
        end
        println("    Basin node $basin_node total rho: $(basin_rho_total)")
        rho_total += basin_rho_total
    end
    return println("Total rho for all basins: $(rho_total)")
end

function test_seed_points()
    sys = with_zip_file("test_data/adamantane_adf.t41", load_data_scm_adf_t41)[1]
    gba_results = fill_with_gps(sys; num_sphere_points=100, seed_points_only=true)
    seed_points = hcat(gba_results["seed_points"]...)
    return plot_system(
        sys;
        cps=gba_results["cps"],
        extra_points=seed_points,
        extra_points_colors=gba_results["cp_ind"],
    )
end

function test_gps()
    # sys = with_zip_file("test_data/h2_adf.t41", load_data_scm_adf_t41)
    sys = with_zip_file(
        "/Users/haiiro/NoSync/CHARGE-3D-0.05.flapw.charge",
        (f) -> load_data_flapw_charge(
            f; T=Float64, z_spacing=0.025, is_periodic=[false, false, false]
        ),
    )[1]
    gba_results = fill_with_gps(sys; num_sphere_points=100, num_non_nuclear_seed_points=0)

    # return gba_results
    return plot_system(
        sys;
        cps=gba_results["cps"],
        gps=gba_results["gps"],
        gps_colors=gba_results["cp_ind"],
    )
    return sys, gba_results
end

function test_plot(sys, gbd)
    seed_points = hcat(gbd["seed_points"]...)
    return plot_system(sys; cps=gbd["cps"], extra_points=seed_points)
end

# test_gps()
# return main();

function test_rho_around_ncps(sys, cp)
    dr = 0.1
    r = cp.r
    @show r
    b = [0, 1]
    for i in b, j in b, k in b
        x = r[1] + i * dr
        y = r[2] + j * dr
        z = r[3] + k * dr
        @show (i, j, k) sys.func_xyz(x, y, z)
    end
end

function check_ncp_neighbor_distances(cps, cpnum)
    r = cps[cpnum].r
    distances = sort([norm(r - cp.r) for cp in cps if cp.type == nuclear_cp])
    for d in distances
        println(d)
    end
end

function generate_grid_points(sys, grid_spacing::Float64)
    # Calculate the number of steps in unit space, based on grid extent and spacing
    steps = [Int.(floor.(norm(v) / grid_spacing)) for v in eachcol(sys.grid.lattice_full)]

    println("Steps: ", steps)

    # Ensure at least one point along each dimension
    steps = max.(steps, 1)

    # Generate indices in unit space (using UnitRange)
    indices = CartesianIndices(steps)

    println("Indices: ", indices)

    # Generate points in unit space
    unit_points = [[idx.I] for idx in indices]

    println("Unit points: ", unit_points)

    # Transform points to the actual grid space
    points = [sys.grid.origin + sys.grid.lattice_full * p for p in unit_points]

    # Reshape for Plotly (assuming 3D)
    X = [p[1] for p in points]
    Y = [p[2] for p in points]
    Z = [p[3] for p in points]

    return X, Y, Z
end

function vol_func(x::Vector{T}) where {T<:AbstractFloat}
    return T(1.0)
end

function debug_inexact_dgb_integration()
    gpi = 1435
    system_gbd_data = open(deserialize, "temp_data/gbd_data.jls")
    gp = system_gbd_data.basin_gbd_data[1].gradient_paths[gpi]
    gp_start_area = system_gbd_data.basin_gbd_data[1].gb_sphere_point_areas[gpi]
    f_list = [system_gbd_data.system.func, vol_func]
    ncps = system_gbd_data.critical_points
    sphere_radius = system_gbd_data.basin_gbd_data[1].sphere_radius
    reverse =
        norm(gp.r[:, end] - ncps[gp.start_cp].r) < norm(gp.r[:, 1] - ncps[gp.start_cp].r)
    int_val = integrate_dgb(
        gp,
        system_gbd_data.system,
        gp_start_area,
        f_list;
        reverse=reverse,
        lower_limit=sphere_radius,
    )
    return nothing
end

function debug_princ_curv()
    system_gbd_data = open(deserialize, "temp_data/gbd_data.jls")
    r_list = [
        [14.232482413889173, 8.58530776466056, 1.6442316687448102],
        [16.106267900215474, 8.561576420503432, 3.4589227708365815],
    ]
    for r in r_list
        principal_curvatures_and_directions(r, system_gbd_data.system)
    end
end

# debug_inexact_dgb_integration()

# main1()
# main(5000);
# gbd_test_restart()
# print_sys_gbd_info(out[1]);
# out;
