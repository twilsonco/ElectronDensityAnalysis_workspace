using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations
include("../test/test_helpers.jl")

sys = with_zip_file("test_data/benzene_adf.t41", load_data_scm_adf_t41)[1]

@info "sys.grid" sys.grid.origin sys.grid.origin .+ sys.grid.lattice_full * ones(3)

# test shrink_grid_bounds
shrunk_grid = shrink_grid_bounds(sys.grid, sys.is_periodic, 0.2)

display(sys.grid)
display(shrunk_grid)
display(sys.grid.origin + sys.grid.lattice_full * ones(3))
display(shrunk_grid.origin + shrunk_grid.lattice_full * ones(3))

# Define grid parameters
grid_num_pts = 100  # Using 100 as in the original testing code

# Create the grid data with extended range
grid = [
    range(
        sys.grid.origin[i] - (sys.grid.lattice_full * 0.5ones(3))[i];  # Start 0.5 lattice units before origin
        stop=sys.grid.origin[i] + (sys.grid.lattice * (Float64.(sys.grid.n_pts) .+ 0.5))[i],  # End 0.5 lattice units after far corner
        length=grid_num_pts,
    ) for i in 1:3
]

# Convert to matrix
r = Matrix(hcat(grid...)')

@info "r matrix size" size(r)

is_periodic = [true, true, true]

# Apply translational symmetry
r_sym = apply_translational_symmetry(r, sys.grid, is_periodic)

println("Original points shape: ", size(r))
println("Symmetry-applied points shape: ", size(r_sym))

# Check which points are in the bounding box
in_box = in_bounding_box(r, sys.grid)
println("Number of points in bounding box before symmetry: ", sum(in_box))
println(
    "Number of points in bounding box after symmetry: ",
    sum(in_bounding_box(r_sym, sys.grid)),
)

# Clamp points to bounding box
r_clamped = clamp_to_bounding_box(r, sys.grid)
println(
    "Number of points in bounding box after clamping: ",
    sum(in_bounding_box(r_clamped, sys.grid)),
)

# Check which points are in the grid
in_grid_result = in_grid(sys.grid, r)
println("Number of points in grid: ", sum(in_grid_result))

# Test with some specific points
test_points = hcat(
    sys.grid.origin,  # Origin
    sys.grid.origin + sum(sys.grid.lattice_full; dims=2),  # Far corner
    sys.grid.origin .+ sys.grid.lattice_full .* [0.5, 0.5, 0.5],  # Middle point
    sys.grid.origin - [0.1, 0.1, 0.1],  # Just outside origin
    sys.grid.origin + sum(sys.grid.lattice_full; dims=2) + [0.1, 0.1, 0.1],  # Just outside far corner
)

println("\nTesting specific points:")
for i in 1:size(test_points, 2)
    point = test_points[:, i]
    println("Point ", i, ": ", point)
    println("  Origin:", sys.grid.origin)
    println("  Far corner:", sys.grid.origin .+ sys.grid.lattice_full * ones(3))
    println("  In bounding box: ", in_bounding_box(point, sys.grid))
    println("  In grid: ", in_grid(sys.grid, point))
    sym_point = apply_translational_symmetry(point, sys.grid, is_periodic)
    println("  After symmetry: ", sym_point)
    println("  Clamped: ", clamp_to_bounding_box(point, sys.grid))
end
