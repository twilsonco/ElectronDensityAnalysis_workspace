using ElectronDensityAnalysis
using BenchmarkTools
using LinearAlgebra
using Random
using PlotlyJS
using Statistics
include("../test/test_helpers.jl")

function generate_seed_points(sys::System, N::Int, spread::Float64=4.0)
    T = eltype(sys.grid.origin)
    center = sys.grid.origin + sys.grid.lattice_full * 0.5ones(3)
    seeds = Matrix{T}(rand(3, N) .- 0.5)  # Random values between -0.5 and 0.5
    seeds .*= spread  # Scale the spread
    seeds .+= center  # Center around the middle of the grid
    bb_gs = shrink_grid_bounds(sys.grid, sys.is_periodic, 0.2)
    clamp_to_bounding_box!(seeds, bb_gs)
    return seeds
end

function test_new_gps()
    # Load the system
    sys = with_zip_file(
        "test_data/adamantane_adf.t41", f -> load_data_scm_adf_t41(f; T=Float32)
    )[1]

    display(sys)

    # Set parameters for streamline tracing
    N = 100  # number of seed points
    max_length = 10.0
    max_steps = 1000

    # Generate seed points near the center of the system

    # Generate seed points
    seeds = generate_seed_points(sys, N)

    cps = find_critical_points(sys)
    cp_dist_threshold = 0.1  # Set this to an appropriate value for your system

    direction = both_dir  # or forward_dir or backward_dir

    gradient_paths = create_gradient_paths(
        seeds,
        sys,
        direction;
        max_length=max_length,
        max_steps=max_steps,
        cps=cps,
        cp_dist_threshold=cp_dist_threshold,
        func_cutoff=0.005,
        term_at_cp=true,
    )

    return plot_system(sys; gps=gradient_paths, cps=cps)
end

# Function to wrap the new parallel method
function new_method(seeds, sys, cps)
    return create_gradient_paths(
        seeds,
        sys,
        both_dir;
        cps=cps,
        max_length=10.0,
        max_steps=1000,
        func_cutoff=0.001,
        term_at_cp=true,
        parallel_execution_num_seeds=size(seeds, 2) + 1,
    )
end

function new_method_par(seeds, sys, cps)
    return create_gradient_paths(
        seeds,
        sys,
        both_dir;
        cps=cps,
        max_length=10.0,
        max_steps=1000,
        func_cutoff=0.001,
        term_at_cp=true,
        parallel_execution_num_seeds=0,
    )
end

function benchmark_new()
    sys = with_zip_file("test_data/adamantane_adf.t41", load_data_scm_adf_t41)[1]
    cps = find_critical_points(sys)
    cp_dist_threshold = 0.1  # Set this to an appropriate value for your system

    # Set parameters for streamline tracing
    N = 1000  # number of seed points

    # Generate seed points
    seeds = generate_seed_points(sys, N)

    # Benchmark the new parallel method
    println("Benchmarking serial method")
    benchmark = @benchmark new_method($seeds, $sys, $cps)
    println("Benchmarking parallel method")
    benchmark_par = @benchmark new_method_par($seeds, $sys, $cps)

    println("Serial benchmark:")
    display(benchmark)

    println("Parallel benchmark:")
    display(benchmark_par)

    return nothing
end

# test_new_gps()

# old_benchmark, new_benchmark = benchmarking_comp()

benchmark_new()
