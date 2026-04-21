using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations
using BenchmarkTools
include("../test/test_helpers.jl")

sys = with_zip_file("test_data/benzene_adf.t41", load_data_scm_adf_t41)[1]

display(sys)
