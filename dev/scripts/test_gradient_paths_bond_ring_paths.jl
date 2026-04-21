using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Quadratic
using BenchmarkTools
include("../test/test_helpers.jl")

sys = with_zip_file("test_data/benzene_adf.t41", load_data_scm_adf_t41)[1]
# sys = with_zip_file("test_data/ethanol_adf.t41", load_data_scm_adf_t41)[1]
# sys = with_zip_file("test_data/adamantane_adf.t41", load_data_scm_adf_t41)[1]
# sys = with_zip_file("test_data/buckyball_adf.t41", load_data_scm_adf_t41)[1]
# sys = with_zip_file("test_data/Ne2_adf.t41", (f) -> load_data_scm_adf_t41(f; T=Float32))[1]
# sys, f3d, f3d_xyz, grad3d, hess3d, func!, grad!, hess! = create_trial_3D_system_interpolated()

gp = create_gradient_path([0.1, 0.1, 0.1], sys, both_dir)
plt = plot_system(sys; gps=[gp])
display(plt)

cps = find_critical_points(sys)
gp = create_gradient_path([0.1, 0.1, 0.1], sys, both_dir; cps=cps)
gp1 = create_gradient_path([0.1, 0.1, 0.1], sys, forward_dir; cps=cps)
gp2 = create_gradient_path([0.1, 0.1, 0.1], sys, backward_dir; cps=cps)
plt = plot_system(sys; gps=[gp], cps=cps)
bond_paths = create_bond_paths(sys, cps)
ring_paths = create_ring_paths(sys, cps)
plt = plot_system(sys; bond_paths=bond_paths, ring_paths=ring_paths, cps=cps)
display(plt)
# remove last CP from list
cps = cps[1:(end - 1)]
bond_paths = create_bond_paths(sys, cps)
ring_paths = create_ring_paths(sys, cps)
plt = plot_system(sys; bond_paths=bond_paths, ring_paths=ring_paths, cps=cps)

display(plt)
