using ElectronDensityAnalysis
using PlotlyJS
include("../test/test_helpers.jl")

ellipsoid_points = points_on_ellipsoid(
    1000, [0.0, 0.0, 0.0], [3.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
)

# show plot 
display(
    PlotlyJS.plot(
        scatter3d(;
            x=ellipsoid_points[:, 1],
            y=ellipsoid_points[:, 2],
            z=ellipsoid_points[:, 3],
            mode="markers",
            marker_size=2.0,
        ),
    ),
)
