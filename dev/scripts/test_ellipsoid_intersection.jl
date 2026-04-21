using ElectronDensityAnalysis
using LinearAlgebra
using PlotlyJS
using Rotations
include("../test/test_helpers.jl")

"""
    ellipsoid_line_segment_intersections(p1, p2, center, λ)

Compute the intersection of a line segment with an ellipsoid defined by the eigenvectors of the Hessian at the cage critical point.

# Arguments
- `p1::Vector{T}`: The first point of the line segment.
- `p2::Vector{T}`: The second point of the line segment.
- `center::Vector{T}`: The center of the ellipsoid.
- `λ::Vector{T}`: The eigenvalues of the Hessian at the cage critical point.

# Returns
A vector of intersection points, which may be empty.
"""
function ellipsoid_line_segment_intersections(
    p1::Vector{T}, p2::Vector{T}, center::Vector{T}, M::Matrix{T}
)::Vector{Vector{T}} where {T<:AbstractFloat}
    # p1, p2 are the points of the line segment
    # center is the center of the ellipsoid
    # M is the matrix of the ellipsoid, where each column is a unit eigenvector of the ellipsoid, 
    # and the magnitude of each eigenvector is λi, where λi is the corresponding eigenvalue of the Hessian at the cage critical point.
    # The line segment is defined by p1 + t(p2 - p1), t in [0,1]
    # The ellipsoid is defined by the equation
    # (p - center)' * M^-1 * (p - center) = 1
    # p is the point on the line segment.
    # The intersection points are the solutions to the equation
    # (p1 + t(p2 - p1) - center)' * M^-1 * (p1 + t(p2 - p1) - center) = 1
    # which is a quadratic equation in t.
    # The solutions are given by the quadratic formula
    # t = (-b ± sqrt(b^2 - 4ac)) / 2a
    # where a = (p2 - p1)' * M^-1 * (p2 - p1)
    # b = 2(p2 - p1)' * M^-1 * (p1 - center)
    # c = (p1 - center)' * M^-1 * (p1 - center) - 1
    # The discriminant is given by
    # D = b^2 - 4ac
    # If D < 0, there are no real solutions, and the line segment does not intersect the ellipsoid.
    # If D = 0, there is one real solution, and the line segment is tangent to the ellipsoid.
    # If D > 0, there are two real solutions, and the line segment intersects the ellipsoid at two points.
    # The intersection points are given by
    # p1 + t1(p2 - p1) and p1 + t2(p2 - p1)
    # where t1 = (-b + sqrt(D)) / 2a and t2 = (-b - sqrt(D)) / 2a.
    # When 0 <= t <= 1, the intersection point is on the line segment.
    Σ = inv(M^2)
    a = (p2 - p1)' * Σ * (p2 - p1)
    b = 2 * (p2 - p1)' * Σ * (p1 - center)
    c = (p1 - center)' * Σ * (p1 - center) - 1
    D = b^2 - 4 * a * c
    out = Vector{Vector{T}}()
    if D == 0
        t = -b / (2 * a)
        if 0 <= t <= 1
            push!(out, p1 + t * (p2 - p1))
        end
    elseif D > 0
        t1 = (-b + sqrt(D)) / (2 * a)
        t2 = (-b - sqrt(D)) / (2 * a)
        if 0 <= t1 <= 1
            push!(out, p1 + t1 * (p2 - p1))
        end
        if 0 <= t2 <= 1
            push!(out, p1 + t2 * (p2 - p1))
        end
    end
    return out
end

# define test ellipsoid lambda values
λ = [4.0, 2.0, 2.0]
M = diagm(λ)

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

# define tangent line segment that intersects along the negative x-axis
p7 = [-λ[1], 1.0, 0.0]
p8 = [-λ[1], -1.0, 0.0]

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
intersections6 = ellipsoid_line_segment_intersections(p7, p8, o, M)
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
for i in [
    intersections1,
    intersections2,
    intersections3,
    intersections4,
    intersections5,
    intersections6,
]
    if !isempty(i)
        push!(
            traces,
            PlotlyJS.scatter3d(;
                x=[j[1] for j in i],
                y=[j[2] for j in i],
                z=[j[3] for j in i],
                mode="markers",
                marker_size=5.0,
            ),
        )
    end
end
for line_seg in [[o, p1], [o, p2], [o, p3], [o, p4], [p7, p8]]
    push!(
        traces,
        PlotlyJS.scatter3d(;
            x=[line_seg[1][1], line_seg[2][1]],
            y=[line_seg[1][2], line_seg[2][2]],
            z=[line_seg[1][3], line_seg[2][3]],
            mode="lines",
        ),
    )
end

layout = Layout(; width=800, height=800, scene_camera_eye=attr(; x=1.25, y=1.25, z=1.25))
PlotlyJS.plot(traces, layout)
