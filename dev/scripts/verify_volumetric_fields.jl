#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using Statistics

# Load ethane system to verify volumetric field computations
include("../test/test_helpers.jl")

println("Loading ethane system...")
sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]

println("\nComputing volumetric curvature fields...")
println("Grid dimensions: $(sys.grid.n_pts)")

# Compute all the curvature fields
println("\n1. Computing mean curvature H(x,y,z)...")
H_field = mean_curvature_field(sys)
println("   Range: [$(minimum(H_field)), $(maximum(H_field))]")
println("   Mean: $(mean(H_field))")
println("   Std: $(std(H_field))")

println("\n2. Computing Gaussian curvature K(x,y,z)...")
K_field = gaussian_curvature_field(sys)
println("   Range: [$(minimum(K_field)), $(maximum(K_field))]")
println("   Mean: $(mean(K_field))")
println("   Std: $(std(K_field))")

println("\n3. Computing RMS curvature...")
rms_field = rms_curvature_field(sys)
println("   Range: [$(minimum(rms_field)), $(maximum(rms_field))]")
println("   Mean: $(mean(rms_field))")
println("   Std: $(std(rms_field))")

println("\n4. Computing shape index S(x,y,z)...")
S_field = shape_index_field(sys)
println("   Range: [$(minimum(S_field)), $(maximum(S_field))]")
println("   Mean: $(mean(S_field))")
println("   Std: $(std(S_field))")

println("\n5. Reference: ρ(x,y,z)...")
println("   Range: [$(minimum(sys.data)), $(maximum(sys.data))]")
println("   Mean: $(mean(sys.data))")
println("   Std: $(std(sys.data))")

# Check if any fields are identical
println("\n=== Checking for duplicate fields ===")

function check_duplicate(name1, field1, name2, field2)
    max_diff = maximum(abs.(field1 .- field2))
    if max_diff < 1e-10
        println("❌ $name1 and $name2 are IDENTICAL (max diff: $max_diff)")
        return true
    else
        println("✓ $name1 and $name2 are DIFFERENT (max diff: $max_diff)")
        return false
    end
end

has_duplicates = false
has_duplicates |= check_duplicate("ρ", sys.data, "H", H_field)
has_duplicates |= check_duplicate("ρ", sys.data, "K", K_field)
has_duplicates |= check_duplicate("ρ", sys.data, "RMS", rms_field)
has_duplicates |= check_duplicate("ρ", sys.data, "S", S_field)
has_duplicates |= check_duplicate("H", H_field, "K", K_field)
has_duplicates |= check_duplicate("H", H_field, "RMS", rms_field)
has_duplicates |= check_duplicate("K", K_field, "RMS", rms_field)

if has_duplicates
    println("\n❌ PROBLEM: Some volumetric fields are duplicates!")
else
    println("\n✓ All volumetric fields are properly independent!")
end
