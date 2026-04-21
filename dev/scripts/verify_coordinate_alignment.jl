using ElectronDensityAnalysis
using LinearAlgebra
include("../test/test_helpers.jl")

function verify_coordinate_alignment()
    println("=== Coordinate Alignment Verification ===")

    # Load real system
    real_sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
    println("Loaded system: $(real_sys.name)")
    println("Grid dimensions: $(real_sys.grid.n_pts)")

    # Generate the same coordinates the TecplotWriter uses
    grid_ranges = generate_interpolation_grid(real_sys.grid)

    println("\n=== Grid Range Information ===")
    for i in 1:3
        println("Dimension $i:")
        println("  Range: $(first(grid_ranges[i])) to $(last(grid_ranges[i]))")
        println("  Length: $(length(grid_ranges[i]))")
        println("  Step: $(step(grid_ranges[i]))")
        println("  Expected length: $(real_sys.grid.n_pts[i])")
    end

    # Test coordinate generation
    imax, jmax, kmax = real_sys.grid.n_pts
    total_points = imax * jmax * kmax

    println("\n=== Coordinate Generation Test ===")
    println("Total expected points: $total_points")

    # Generate coordinates the same way TecplotWriter does
    x_coords = Float32[]
    y_coords = Float32[]
    z_coords = Float32[]

    for k in 1:kmax, j in 1:jmax, i in 1:imax
        push!(x_coords, Float32(grid_ranges[1][i]))
        push!(y_coords, Float32(grid_ranges[2][j]))
        push!(z_coords, Float32(grid_ranges[3][k]))
    end

    println("Generated coordinate arrays:")
    println("  X coords: $(length(x_coords)) points")
    println("  Y coords: $(length(y_coords)) points")
    println("  Z coords: $(length(z_coords)) points")

    # Test data ordering
    println("\n=== Data Ordering Test ===")
    data_size = size(real_sys.data)
    println("System data size: $data_size")

    # Generate data the same way TecplotWriter does
    data_ijk = zeros(Float32, total_points)
    idx = 1
    for k in 1:kmax, j in 1:jmax, i in 1:imax
        data_ijk[idx] = Float32(real_sys.data[i, j, k])
        idx += 1
    end

    println("Generated data array length: $(length(data_ijk))")
    println("Data statistics:")
    println("  Min: $(minimum(data_ijk))")
    println("  Max: $(maximum(data_ijk))")
    println("  Mean: $(sum(data_ijk) / length(data_ijk))")

    # Compare with original data
    orig_min = minimum(real_sys.data)
    orig_max = maximum(real_sys.data)
    orig_mean = sum(real_sys.data) / length(real_sys.data)

    println("Original data statistics:")
    println("  Min: $orig_min")
    println("  Max: $orig_max")
    println("  Mean: $orig_mean")

    # Check alignment
    println("\n=== Alignment Verification ===")

    # Test a few specific points
    test_indices = [(1, 1, 1), (imax÷2, jmax÷2, kmax÷2), (imax, jmax, kmax)]

    for (i, j, k) in test_indices
        # Calculate linear index in IJK order
        linear_idx = (k-1)*imax*jmax + (j-1)*imax + i

        println("Point ($i, $j, $k):")
        println("  Linear index: $linear_idx")
        println(
            "  Coordinates: ($(grid_ranges[1][i]), $(grid_ranges[2][j]), $(grid_ranges[3][k]))",
        )
        println("  Data value: $(real_sys.data[i, j, k])")
        println("  IJK array value: $(data_ijk[linear_idx])")
        println("  Match: $(real_sys.data[i, j, k] ≈ data_ijk[linear_idx])")
        println()
    end

    # Overall data match
    all_match = all(
        real_sys.data[i, j, k] ≈ data_ijk[(k - 1) * imax * jmax + (j - 1) * imax + i] for
        k in 1:kmax, j in 1:jmax, i in 1:imax
    )

    if all_match
        println("✅ All data points match between original and IJK-ordered arrays!")
    else
        println("❌ Data mismatch detected!")
    end

    println("\n🎉 Coordinate alignment verification complete!")

    return (x_coords, y_coords, z_coords, data_ijk)
end

# Run verification
coords_data = verify_coordinate_alignment()
