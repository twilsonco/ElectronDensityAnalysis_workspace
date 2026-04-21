using Random, Clustering, PlotlyJS, Statistics, LinearAlgebra

# Function to generate synthetic clustered 3D point cloud data
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
spacing_factor = 0.2  # 0.4 ensures points are about 80% of the way to neighboring clusters

# Generate a realistic clustered point cloud
points = generate_point_cloud(points_per_cluster, corner_value, spacing_factor)

# Convert points to a matrix format required by Clustering.jl
points_matrix = hcat(points...)

# Define a range of k values to test
k_values = 1:15

# Initialize an array to store the within-cluster sum of squares (WCSS) for each k
wcss_values = Float64[]

# Perform k-means clustering for each k value and calculate WCSS
for k in k_values
    clusters = kmeans(points_matrix, k)
    wcss = clusters.totalcost
    push!(wcss_values, wcss)
end

# Plot the elbow curve
elbow_plot = PlotlyJS.plot(
    scatter(; x=k_values, y=wcss_values, mode="lines+markers"),
    Layout(;
        title="Elbow Method for Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="WCSS",
    ),
)

# Display the elbow plot
display(elbow_plot)

# Optional: Visualize the clustered point cloud with the optimal k
optimal_k = 8  # Replace this with the k value you determine from the elbow plot
optimal_clusters = kmeans(points_matrix, optimal_k)

# Assign colors to each cluster for visualization knowing that they're in the points_matrix
# in order of the clusters.
cluster_colors = repeat(1:optimal_k; inner=points_per_cluster)

# Create the 3D scatter plot
scatter_plot = PlotlyJS.plot(
    scatter3d(;
        x=points_matrix[1, :],
        y=points_matrix[2, :],
        z=points_matrix[3, :],
        mode="markers",
        marker=attr(; color=cluster_colors, size=5),
    ),
    Layout(;
        title="3D Point Cloud TRUE Clustering",
        scene=attr(; xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    ),
)

# Display the scatter plot
display(scatter_plot)

# One more 3d scatter plot showing coloring based on the kmeans clustering
# Associate true clusters with kmeans clusters based on cluster centers
# Calculate the distance between each true cluster center and each kmeans cluster center
# and put in a dictionary so that we can use the same colors for the same clusters.

# Calculate true cluster centers
true_cluster_centers = [
    mean(
        points_matrix[:, ((i - 1) * points_per_cluster + 1):(i * points_per_cluster)];
        dims=2,
    ) for i in 1:optimal_k
]

# Calculate distances between true cluster centers and k-means cluster centers
kmeans_cluster_centers = optimal_clusters.centers
distances = [
    norm(true_center .- kmeans_center) for
    true_center in true_cluster_centers, kmeans_center in eachcol(kmeans_cluster_centers)
]

# Map true clusters to k-means clusters based on minimum distance
cluster_mapping = Dict(i => argmin(distances[:, i]) for i in 1:optimal_k)

# Assign colors to each cluster based on the k-means clustering
kmeans_cluster_colors = [
    cluster_mapping[optimal_clusters.assignments[i]] for
    i in 1:length(optimal_clusters.assignments)
]

# Create the 3D scatter plot
scatter_plot2 = PlotlyJS.plot(
    scatter3d(;
        x=points_matrix[1, :],
        y=points_matrix[2, :],
        z=points_matrix[3, :],
        mode="markers",
        marker=attr(; color=kmeans_cluster_colors, size=5),
    ),
    Layout(;
        title="3D Point Cloud K-Means Clustering",
        scene=attr(; xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    ),
)

# Display the scatter plot
display(scatter_plot2)

# Show the cluster mapping for debugging
@show cluster_mapping
