using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Quadratic
using BenchmarkTools
include("../test/test_helpers.jl")

sys = with_zip_file(
    "$DATA_DIR/Cu_fcc_PBE_TZ2P_AE_ZORA_periodic-xyz_band.t41", load_data_scm_band_t41
)[1]
write_data_cube(sys, "$DATA_DIR/Cu_fcc_PBE_TZ2P_AE_ZORA_periodic-xyz_band.cub")
