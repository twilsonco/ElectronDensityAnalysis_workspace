using ElectronDensityAnalysis
using PlotlyJS
include("../test/test_helpers.jl")

sphere_points = points_on_sphere_regular(1000, [0.0, 0.0, 0.0], 1.0)

# show plot 
display(
    PlotlyJS.plot(
        scatter3d(;
            x=sphere_points[:, 1],
            y=sphere_points[:, 2],
            z=sphere_points[:, 3],
            mode="markers",
            marker_size=2.0,
        ),
    ),
)
