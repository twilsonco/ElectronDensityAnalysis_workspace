#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using ElectronDensityAnalysis.TecplotWriter
using Interpolations: Linear

# Include test helpers (for with_zip_file, handle_checkpoint)
include("../test/test_helpers.jl")

function export_ethane_full_sgbd(; cell_centered_spheres::Bool=false)
    println(
        "Exporting full SystemGBDData for ethane (cell_centered_spheres=$(cell_centered_spheres))...",
    )

    checkpoint_dir = "/Volumes/HaiiroStudio_Ext_6T/gbd_checkpoints/Ethane"
    load_sys_args = (
        "test_data/ethane_adf.t41", (f) -> with_zip_file(f, load_data_scm_adf_t41)
    )

    sys_checkpoint_file = joinpath(checkpoint_dir, "system.jls")
    sys = handle_checkpoint(sys_checkpoint_file, with_zip_file, load_sys_args)[1]

    sgbd = gradient_bundle_decomposition(
        sys;
        num_gbs=5000,
        interp_type=Linear,
        func_cutoff=1e-3,
        curvature_threshold=0.8 * 0.5π,
        atom_bounding_box_shrink_factor=0.8,
        checkpoint_dir=checkpoint_dir,
        system_dependency_files=[sys_checkpoint_file],
    )

    println("Critical points: $(length(sgbd.critical_points))")
    println("DGBs: $(length(sgbd.differential_gradient_bundles))")
    println("Rejected paths: $(length(sgbd.rejected_gradient_paths))")
    println("Atom spheres: $(length(sgbd.atom_sphere_data))")
    println("Functions: $(sgbd.function_names)")

    out_file = if cell_centered_spheres
        "ethane_full_sgbd_cellcentered.plt"
    else
        "ethane_full_sgbd_nodal.plt"
    end

    # Optional export of triangulated condensed basin boundary surfaces via env vars
    boundary_surfaces = get(ENV, "GBD_BOUNDARY_SURFACES", "0") == "1"
    boundary_samples = try
        parse(Int, get(ENV, "GBD_BOUNDARY_SAMPLES", "50"))
    catch
        50
    end

    # Optional export of DGB gradient paths via env var
    print_dgbs = get(ENV, "GBD_PRINT_DGBS", "1") == "1"

    # Optional export of per-basin FE_Triangle patches on spheres via env var
    basin_spheres = get(ENV, "GBD_SPHERE_BASINS", "0") == "1"

    TecplotWriter.write_sgbd_plt_file(
        out_file,
        "Ethane",
        sgbd;
        title="Ethane full SystemGBDData export",
        system=sys,
        cell_centered_spheres=cell_centered_spheres,
        print_condensed_basin_boundary_gps=boundary_surfaces,
        boundary_surface_num_samples=boundary_samples,
        print_condensed_basin_spheres=basin_spheres,
        print_dgbs=print_dgbs,
    )

    println("Wrote: $(out_file)")
end

# if abspath(PROGRAM_FILE) == @__FILE__
cell_centered = get(ENV, "GBD_CELL_CENTERED", "0") == "1"
export_ethane_full_sgbd(; cell_centered_spheres=cell_centered)
# end
