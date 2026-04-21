using ElectronDensityAnalysis
using LinearAlgebra
using PlotlyJS
using Rotations
include("../test/test_helpers.jl")

# define test ellipsoid lambda values
λ = [4.0, 2.0, 2.0]
M = diagm(λ)

# rotate M pi/4 radians about the z-axis using the Rotations package
α = π / 4

r = AngleAxis(α, 0.0, 0.0, 1.0)
q = QuatRotation(r)
M = q * M * q'

# rotate M pi/4 radians about the y-axis using the Rotations package
r = AngleAxis(α, 0.0, 1.0, 0.0)
q = QuatRotation(r)
M = q * M * q'

# compute ellipsoid points
ellipsoid_points = points_on_ellipsoid(5000, [0.0, 0.0, 0.0], M)
# get max xyz values of ellipsoid points
max_xyz = maximum(ellipsoid_points; dims=1)
@show max_xyz

# four intersecting line segments, one along each axis, and
# one in the 1,1,1 direction
o = [0.0, 0.0, 0.0]
p1 = [6.0, 0.0, 0.0]
p2 = [0.0, 3.0, 0.0]
p3 = [0.0, 0.0, 3.0]
p4 = [3.0, 3.0, 3.0]

# define non-intersecting line segment
p5 = [8.0, 0.0, 0.0]
p6 = [0.0, 0.0, 4.0]

# compute intersection points
intersections1 = ellipsoid_line_segment_intersections(o, p1, o, M)
@show intersections1
intersections2 = ellipsoid_line_segment_intersections(o, p2, o, M)
@show intersections2
intersections3 = ellipsoid_line_segment_intersections(o, p3, o, M)
@show intersections3
intersections4 = ellipsoid_line_segment_intersections(o, p4, o, M)
@show intersections4
intersections5 = ellipsoid_line_segment_intersections(p5, p6, o, M)
@show intersections5

# Now test 
# function ellipsoid_path_intersection(
# path_r::Matrix{T}, cp_r::Vector{T}, M::Matrix{T}
# )::Vector{Float64} where {T<:AbstractFloat}
# We'll take the 0 -> p1 path and make a 50 point path out of it
path_r = [range(0.0, p1[1]; length=50)'; zeros(50)'; zeros(50)']
cp_r = o
intersections6 = ellipsoid_path_intersection(path_r, cp_r, M)
@show intersections6

# plot ellipsoid and line segments, and intersection points

traces = [
    PlotlyJS.scatter3d(;
        x=ellipsoid_points[:, 1],
        y=ellipsoid_points[:, 2],
        z=ellipsoid_points[:, 3],
        mode="markers",
        marker_size=2.0,
    ),
]
for i in [intersections1, intersections2, intersections3, intersections4, intersections5]
    if !isempty(i)
        push!(
            traces,
            PlotlyJS.scatter3d(;
                x=[j[1] for j in i],
                y=[j[2] for j in i],
                z=[j[3] for j in i],
                mode="markers",
                marker_size=10.0,
            ),
        )
    end
end
for line_seg in [[o, p1], [o, p2], [o, p3], [o, p4]]
    push!(
        traces,
        PlotlyJS.scatter3d(;
            x=[line_seg[1][1], line_seg[2][1]],
            y=[line_seg[1][2], line_seg[2][2]],
            z=[line_seg[1][3], line_seg[2][3]],
            mode="lines",
            line_width=5.0,
        ),
    )
end

layout = Layout(; width=800, height=800, scene_camera_eye=attr(; x=1.25, y=1.25, z=1.25))
PlotlyJS.plot(traces, layout)
