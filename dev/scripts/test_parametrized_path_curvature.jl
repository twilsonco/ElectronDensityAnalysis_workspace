using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Quadratic
using BenchmarkTools
using Statistics: mean, median
include("../test/test_helpers.jl")

sys = with_zip_file("$DATA_DIR/benzene_adf.t41", load_data_scm_adf_t41)[1]

gp = create_gradient_path([0.1, 0.1, 0.1], sys, both_dir)

pgp_len = gp_parametrize(gp; f="length")
pgp_curv = gp_parametrize_curvature(pgp_len)

# compute 100 points using pgp_len and corresponding curvatures
x_range = range(pgp_len.fmin + 0.1pgp_len.fmax, pgp_len.fmax - 0.1pgp_len.fmax; length=1000)
r = hcat([pgp_len[t] for t in x_range]...)
curv = [pgp_curv[t] for t in x_range]
@show maximum(curv)
# normalize curv to be between 1 and 10
curv10 = 3 .+ 20 .* (curv .- minimum(curv)) ./ (maximum(curv) - minimum(curv))

plt = plot_system(sys; extra_points=r, extra_points_colors=curv, extra_points_sizes=curv10)
