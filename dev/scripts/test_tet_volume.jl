using ElectronDensityAnalysis
using PlotlyJS
include("../test/test_helpers.jl")

p = [0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]

# tet_volume takes a 3x4 matrix
vol = tet_volume(p)

println("Volume of tetrahedron: ", vol, ", which should be 1/6.")
