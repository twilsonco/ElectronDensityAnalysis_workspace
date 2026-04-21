using ElectronDensityAnalysis
using PlotlyJS
using Delaunay
include("../test/test_helpers.jl")

# Usage
points = points_on_sphere_regular(1000, [0.0, 0.0, 0.0], 1.0)
mesh = delaunay(points)

hull = mesh.convex_hull
elem_connectivity = get_triangulation_convex_hull_element_connectivity(
    hull; single_shared_node_neighbors=true
)

elem1_neighbors = elem_connectivity[1]
elem1_nodes = hull[1, :]
neighbor_nodes = hull[elem1_neighbors, :]

@show elem1_neighbors elem1_nodes neighbor_nodes

node_connectivity = get_triangulation_convex_hull_node_connectivity(hull)

@show size(node_connectivity) size(node_connectivity[1])

node1_neighbors = node_connectivity[hull[1, 1]]
@show node1_neighbors
