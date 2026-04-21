using Random, Clustering, PlotlyJS, Statistics, LinearAlgebra, Distances

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

# Function to calculate silhouette score
function silhouette_score(points_matrix, labels)
    n = size(points_matrix, 2)
    a = zeros(n)
    b = zeros(n)
    for i in 1:n
        same_cluster = [j for j in 1:n if labels[j] == labels[i] && j != i]
        other_clusters = [j for j in 1:n if labels[j] != labels[i]]
        if length(same_cluster) > 0
            a[i] = mean([
                euclidean(points_matrix[:, i], points_matrix[:, j]) for j in same_cluster
            ])
        else
            a[i] = 0.0
        end
        b[i] = minimum([
            mean([
                euclidean(points_matrix[:, i], points_matrix[:, j]) for
                j in other_clusters if labels[j] == k
            ]) for k in unique(labels) if k != labels[i]
        ])
    end
    s = (b .- a) ./ max.(a, b)
    return mean(s)
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
k_values = 2:15  # Silhouette score is undefined for k=1

# Initialize an array to store the silhouette scores for each k
silhouette_scores = Float64[]

# Perform k-means clustering for each k value and calculate silhouette score
for k in k_values
    clusters = kmeans(points_matrix, k)
    labels = clusters.assignments
    silhouette_score_value = silhouette_score(points_matrix, labels)
    push!(silhouette_scores, silhouette_score_value)
end

# Find the optimal k based on the maximum silhouette score
optimal_k = k_values[argmax(silhouette_scores)]
println("The optimal k is: $optimal_k")

# Plot the silhouette scores
silhouette_plot = PlotlyJS.plot(
    scatter(; x=k_values, y=silhouette_scores, mode="lines+markers"),
    Layout(;
        title="Silhouette Scores for Different k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
    ),
)

# Display the silhouette plot
display(silhouette_plot)
