using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Quadratic
using BenchmarkTools
using Statistics: mean, median
include("../test/test_helpers.jl")

sys = with_zip_file("test_data/benzene_adf.t41", load_data_scm_adf_t41)[1]
# sys = with_zip_file("test_data/ethanol_adf.t41", load_data_scm_adf_t41)
# sys = with_zip_file("test_data/adamantane_adf.t41", load_data_scm_adf_t41)
# sys = with_zip_file("test_data/buckyball_adf.t41", load_data_scm_adf_t41)
# sys = with_zip_file("test_data/Ne2_adf.t41", (f) -> load_data_scm_adf_t41(f; T=Float32))
# sys, f3d, f3d_xyz, grad3d, hess3d, func!, grad!, hess! = create_trial_3D_system_interpolated()

cps = find_critical_points(sys)

# test principal curvatures and directions calculation
r = [0.1, 0.1, 0.1]
vals, vecs = principal_curvatures_and_directions(r, sys)
@show vals vecs

k2 = isosurface_2x_mean_curvature_at_point(r, sys)
@show k2

gp = create_gradient_path([0.1, 0.1, 0.1], sys, both_dir; cps=cps)
@show size(gp.r, 2)
# plt = plot_system(sys; gps=[gp])

path_point_distances = [norm(gp.r[:, i] - gp.r[:, i - 1]) for i in 2:size(gp.r, 2)]
min_point_distance = minimum(path_point_distances)
max_point_distance = maximum(path_point_distances)
mean_point_distance = mean(path_point_distances)
median_point_distance = median(path_point_distances)
gp_length = path_length(gp.r)
# point spacing is the greater of mean or median
point_spacing = max(1e-6, min(mean_point_distance, median_point_distance))
num_points_gp = Int(ceil(gp_length / point_spacing))
@show min_point_distance max_point_distance mean_point_distance median_point_distance gp_length point_spacing num_points_gp

# error("stop here")

pgp = gp_parametrize(gp; interp_type=Linear)

pgp_2H = gp_parametrize_2H(pgp, sys; gp=gp)
@info "benchmark gp_parametrize_dA with pgp_2H"
@btime gp_parametrize_dA(pgp, pgp_2H, 0.01; gp=gp)
@info "benchmark gp_parametrize_dA without pgp_2H"
@btime gp_parametrize_dA(
    pgp, r -> isosurface_2x_mean_curvature_at_point(pgp[r], sys), 0.01; gp=gp
)

# loop to check error convergence of gp_parametrize_2H vs num_points
for num_points in [50, 100, 500, 1000, 5000, 10000, num_points_gp]
    local pgp_2H = gp_parametrize_2H(pgp, sys; gp=gp, num_points=num_points)
    local pgp_dA = gp_parametrize_dA(pgp, pgp_2H, 0.01; gp=gp, num_points=num_points)
    local pgp_dA1 = gp_parametrize_dA(
        pgp,
        r -> isosurface_2x_mean_curvature_at_point(pgp[r], sys),
        0.01;
        gp=gp,
        num_points=num_points,
    )
    # local pgp_dA = gp_parametrize_dA(pgp, sys, 0.01; num_points=num_points)
    pgp_norms = [
        norm(gp.r[:, i] - pgp[path_length(gp.r; end_ind=i)]) for i in 2:size(gp.r, 2)
    ]
    max_pgp_norm = maximum(pgp_norms)
    pgp_2H_val_diffs = [
        abs(
            pgp_2H(path_length(gp.r; end_ind=i)) -
            isosurface_2x_mean_curvature_at_point(gp.r[:, i], sys),
        ) for i in 2:size(gp.r, 2)
    ]
    max_2H_diff = maximum(pgp_2H_val_diffs)
    pgp_dA_val_diffs = [
        abs(pgp_dA(path_length(gp.r; end_ind=i)) - pgp_dA1(path_length(gp.r; end_ind=i)))
        for i in 2:size(gp.r, 2)
    ]
    max_dA_diff = maximum(pgp_dA_val_diffs)
    @show num_points max_pgp_norm max_2H_diff max_dA_diff
end
