using ElectronDensityAnalysis
using PlotlyJS
# using Delaunay
using Distributed
include("../test/test_helpers.jl")

# Step 1: Set up distributed computing
# Add worker processes (adjust the number as needed)
addprocs(4)

# Step 2: Load necessary packages on all processes
@everywhere using Delaunay: delaunay

# Usage
points_sets = [points_on_sphere_regular(1000 + i * 200, [0.0, 0.0, 0.0], 1.0) for i in 1:4]
num_iterations = 10
for p in points_sets
    @show size(p)
    @show eltype(p)
end
meshes = pmap(delaunay, points_sets)

# print info about each mesh
for (i, mesh) in enumerate(meshes)
    println("Mesh $i:")
    println("  Number of vertices: $(size(mesh.points, 2))")
    println("  Number of simplices: $(size(mesh.simplices, 2))")
    # print min, mean, max, std triangle areas

end
