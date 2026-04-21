#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using ElectronDensityAnalysis
using Statistics

# Test that volumetric curvature fields are different from ρ
include("../test/test_helpers.jl")

println("\n=== Verifying Volumetric Variable Independence ===\n")

# Load a simple system
println("Loading ethane system...")
sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]

# Get the electron density field
rho = sys.data

# Compute curvature fields
println("Computing curvature fields...")
H_field = mean_curvature_field(sys)
K_field = gaussian_curvature_field(sys)
rms_field = rms_curvature_field(sys)
S_field = shape_index_field(sys)

println("\n=== Field Statistics ===")
println("ρ (electron density):")
println("  min: $(minimum(rho)), max: $(maximum(rho)), mean: $(mean(rho))")

println("\nH (mean curvature):")
println("  min: $(minimum(H_field)), max: $(maximum(H_field)), mean: $(mean(H_field))")

println("\nK (Gaussian curvature):")
println("  min: $(minimum(K_field)), max: $(maximum(K_field)), mean: $(mean(K_field))")

println("\nRMS curvature:")
println(
    "  min: $(minimum(rms_field)), max: $(maximum(rms_field)), mean: $(mean(rms_field))"
)

println("\nS (shape index):")
println("  min: $(minimum(S_field)), max: $(maximum(S_field)), mean: $(mean(S_field))")

# Check that fields are different
println("\n=== Independence Tests ===")

# First check if fields are numerically identical
function check_identical(field1, field2, name1, name2)
    # Flatten arrays
    f1 = field1[:]
    f2 = field2[:]

    # Check if arrays are exactly equal
    max_diff = maximum(abs.(f1 .- f2))

    println("\nComparing $name1 and $name2:")
    println("  Max absolute difference: $(max_diff)")

    if max_diff < 1e-10
        println("  ✗ Fields are IDENTICAL (duplicates)")
        return false
    else
        # Calculate correlation coefficient
        corr = cor(f1, f2)
        println("  Correlation: $(round(corr, digits=4))")

        if abs(corr) > 0.95
            println("  ⚠️  Fields are highly correlated (but not identical)")
        else
            println("  ✓ Fields are independent")
        end
        return true
    end
end

all_good = true
all_good &= check_identical(rho, H_field, "ρ", "H")
all_good &= check_identical(rho, K_field, "ρ", "K")
all_good &= check_identical(rho, rms_field, "ρ", "RMS")
all_good &= check_identical(rho, S_field, "ρ", "S")
# H and K are naturally highly correlated (both derived from principal curvatures)
# but they should NOT be identical
println("\nChecking H vs K (expect high correlation but not identical):")
all_good &= check_identical(H_field, K_field, "H", "K")

println("\n=== Result ===")
if all_good
    println("✓ All volumetric fields are properly independent!")
else
    println("✗ Some fields appear to be duplicates!")
    exit(1)
end
