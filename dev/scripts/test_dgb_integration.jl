using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Quadratic
using BenchmarkTools
include("../test/test_helpers.jl")

sys = with_zip_file("$DATA_DIR/benzene_adf.t41", load_data_scm_adf_t41)[1]

cps = find_critical_points(sys)
r = 0.2
N = 1000
# get 1/N the sphere area for a sphere of radius r
a_0 = 4 * pi * r^2 / N
p = sys.atoms[1].r
p[1] += 0.2
gp = create_gradient_path(p, sys, backward_dir; cps=cps)
f_list = [r -> 1.0, r -> sys.func(r)]
int_vals = integrate_dgb(gp, sys, a_0, f_list)
