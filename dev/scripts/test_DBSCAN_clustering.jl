using Clustering
using PlotlyJS
using Random
using Distances

# Function to generate a point cloud with 8 distinct clusters
function generate_point_cloud(
    points_per_cluster::Int, corner_value::Float64, spacing_factor::Float64
)
    points = []
    Random.seed!(1234)  # For reproducibility

    # Define the corners of a cube with the given corner value
    cube_corners = [
        [0, 0, 0],
        [0, 0, corner_value],
        [0, corner_value, 0],
        [0, corner_value, corner_value],
        [corner_value, 0, 0],
        [corner_value, 0, corner_value],
        [corner_value, corner_value, 0],
        [corner_value, corner_value, corner_value],
    ]

    for i in 1:8
        center = cube_corners[i]
        # Generate points around the cluster center
        for _ in 1:points_per_cluster
            point = center .+ randn(3) .* (corner_value * spacing_factor)
            push!(points, point)
        end
    end
    return points
end

# Parameters
points_per_cluster = 300
corner_value = 10.0
spacing_factor = 0.1  # Adjusted spacing factor

# Generate a realistic clustered point cloud
points = hcat(generate_point_cloud(points_per_cluster, corner_value, spacing_factor)...)

@show size(points)

# Precompute the distance matrix
# distance_matrix = pairwise(Euclidean(), points_matrix)

# @show size(distance_matrix)

# Perform DBSCAN clustering
eps = 2.0
min_samples = 5
result = dbscan(points, eps)

# Extract cluster labels from the result
labels = result.assignments

@show length(result.clusters)

# Print the cluster labels for each point
# println("Cluster labels:")
# println(labels)

# Create the 3D scatter plot
scatter_plot = PlotlyJS.plot(
    scatter3d(;
        x=points[1, :],
        y=points[2, :],
        z=points[3, :],
        mode="markers",
        marker_color=labels,
        marker_size=3,
        marker_opacity=0.8,
    ),
    Layout(;
        title="DBSCAN Clustering of 3D Point Cloud",
        scene=attr(; camera_eye=attr(; x=1.25, y=1.25, z=1.25)),
    ),
)

# Display the scatter plot
display(scatter_plot)
