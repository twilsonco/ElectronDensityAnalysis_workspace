using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations: Quadratic
using BenchmarkTools
using PlotlyJS
include("../test/test_helpers.jl")

cone_points = points_on_double_cone(
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    pi / 4,
    pi / 8,
    20000,
    1000,
    1.0,
)

# show plot 
display(
    PlotlyJS.plot(
        scatter3d(;
            x=cone_points[:, 1],
            y=cone_points[:, 2],
            z=cone_points[:, 3],
            mode="markers",
            marker_size=2.0,
        ),
    ),
)
