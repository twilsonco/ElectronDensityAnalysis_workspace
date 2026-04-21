using ElectronDensityAnalysis
using LinearAlgebra
using Interpolations
using BenchmarkTools
include("../test/test_helpers.jl")

sys = with_zip_file("test_data/benzene_adf.t41", load_data_scm_adf_t41)[1]

# Define grid parameters
grid_num_pts = 100  # Using 100 as in the original testing code

# Create the grid data
grid = [
    range(
        sys.grid.origin[i] + (sys.grid.lattice * 2ones(3))[i];
        stop=sys.grid.origin[i] + (sys.grid.lattice * Float64.(sys.grid.n_pts))[i] -
             (sys.grid.lattice * 2ones(3))[i],
        length=grid_num_pts,
    ) for i in 1:3
]

# Convert to matrix
r = Matrix(hcat(grid...)')

@info "r matrix size" size(r)

# Testing single point
g = sys.grad(r[:, 1])
h = sys.hess(r[:, 1])
func = sys.func(r[:, 1])
func_xyz = sys.func_xyz(r[:, 1]...)

@info "Results from using r as a 3x1 vector"
@show r[:, 1] func_xyz func g h

# Testing multiple points
func_xyz_multi = sys.func_xyz.(r[1, :], r[2, :], r[3, :])
g_multi = sys.grad(r)
h_multi = sys.hess(r)
func_multi = sys.func(r)

@info "Results from using r as a 3xN matrix" func_multi g_multi h_multi

# Testing in-place versions
F = zeros(grid_num_pts)
sys.func!(F, r)

# Change G to be a vector of vectors
G = [zeros(3) for _ in 1:grid_num_pts]
sys.grad!(G, r)

# Change J to be a vector of matrices
J = [zeros(3, 3) for _ in 1:grid_num_pts]
sys.hess!(J, r)

# Verifying results
@assert func_multi ≈ F
@assert g_multi ≈ G
@assert h_multi ≈ J

@show F[1] G[1] J[1]

# Benchmarking functions
function bench_func_xyz(sys, r)
    return sys.func_xyz.(r[1, :], r[2, :], r[3, :])
end

function bench_func(sys, r)
    return sys.func(r)
end

function bench_grad(sys, r)
    return sys.grad(r)
end

function bench_hess(sys, r)
    return sys.hess(r)
end

function bench_func!(sys, out, r)
    return sys.func!(out, r)
end

function bench_grad!(sys, out, r)
    return sys.grad!(out, r)
end

function bench_hess!(sys, out, r)
    return sys.hess!(out, r)
end

# Run benchmarks
b_func_xyz = @benchmark bench_func_xyz($sys, $r)
b_func = @benchmark bench_func($sys, $r)
b_grad = @benchmark bench_grad($sys, $r)
b_hess = @benchmark bench_hess($sys, $r)
b_func! = @benchmark bench_func!($sys, $F, $r)
b_grad! = @benchmark bench_grad!($sys, $G, $r)
b_hess! = @benchmark bench_hess!($sys, $J, $r)

# Print results
println("Benchmark results:")
println("sys.func_xyz.(r[1, :], r[2, :], r[3, :]):")
display(b_func_xyz)
println("\nsys.func(r):")
display(b_func)
println("\nsys.grad(r):")
display(b_grad)
println("\nsys.hess(r):")
display(b_hess)
println("\nsys.func!(out, r):")
display(b_func!)
println("\nsys.grad!(out, r):")
display(b_grad!)
println("\nsys.hess!(out, r):")
display(b_hess!)

# Compare median times
function print_comparison(name1, b1, name2, b2)
    t1 = median(b1.times)
    t2 = median(b2.times)
    println("\nComparison: $name1 vs $name2")
    println("$name1 median time: $(t1) ns")
    println("$name2 median time: $(t2) ns")
    return println("Speedup factor: $(t1 / t2)")
end

print_comparison(
    "sys.func_xyz(r[1, :], r[2, :], r[3, :])", b_func_xyz, "sys.func(r)", b_func
)
print_comparison("sys.func(r)", b_func, "sys.func!(out, r)", b_func!)
print_comparison("sys.grad(r)", b_grad, "sys.grad!(out, r)", b_grad!)
print_comparison("sys.hess(r)", b_hess, "sys.hess!(out, r)", b_hess!)
