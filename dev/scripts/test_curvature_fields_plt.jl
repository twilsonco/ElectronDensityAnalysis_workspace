using ElectronDensityAnalysis
using LinearAlgebra

println("Testing curvature field export to Tecplot PLT file\n")

# Create a simple test system with a spherical Gaussian density
function gaussian_3d(x, y, z)
    r2 = x^2 + y^2 + z^2
    return exp(-r2)
end

# Create a small test grid (use a reasonable size for visualization)
n_pts = [21, 21, 21]
origin = [-2.0, -2.0, -2.0]
lattice = [0.2 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.2]

println("Creating grid: $(n_pts) points")
gs = GridSpec(origin, lattice, n_pts)

# Generate the density data
println("Generating Gaussian density data...")
data = zeros(n_pts...)
for k in 1:n_pts[3]
    for j in 1:n_pts[2]
        for i in 1:n_pts[1]
            r = origin + lattice * [i-1, j-1, k-1]
            data[i, j, k] = gaussian_3d(r[1], r[2], r[3])
        end
    end
end

# Create interpolated fields
func_xyz, func, grad, hess, func!, grad!, hess! = ElectronDensityAnalysis.create_interpolated_fields(
    data, gs
)

# Create the system
atoms = [NuclearCoordinate([0.0, 0.0, 0.0], 1)]  # Hydrogen at origin
system = System(
    "Gaussian Density Test",
    "test",
    atoms,
    gs,
    data,
    [false, false, false],
    func_xyz,
    func,
    grad,
    hess,
    func!,
    grad!,
    hess!,
)

println("System created:")
println(system)
println()

# Define the output filename
output_file = "test_curvature_fields.plt"
println("Writing PLT file: $output_file")

# Define variables including the new curvature fields
variables = [
    "X",
    "Y",
    "Z",
    "Electron Density",
    "Mean Curvature",
    "Gaussian Curvature",
    "RMS Curvature",
    "Shape Index",
]

println("Variables to be written:")
for var in variables
    println("  - $var")
end
println()

# Open PLT file and write the system
TecplotWriter.open_plt_file(
    output_file, "Curvature Fields Test", join(variables, ","); scratchdir="."
)

# Write the system zone with all curvature fields
println("Writing system zone with curvature fields...")
TecplotWriter.write_zone(
    "Gaussian System",
    system;
    system=system,
    variables=variables,
    aux_data=[
        "Description=Test of isosurface curvature fields", "Method=Gradient Bundle Analysis"
    ],
)

# Close the file
TecplotWriter.close_plt_file()

println("\n✓ Successfully created PLT file with curvature fields!")
println("  File: $output_file")
println("  Size: $(filesize(output_file)) bytes")
println("\nYou can now open this file in Tecplot 360 to visualize:")
println("  - Mean Curvature (H) - average of principal curvatures")
println("  - Gaussian Curvature (K) - product of principal curvatures")
println("  - RMS Curvature - root mean square of principal curvatures")
println(
    "  - Shape Index (S) - classifies local surface shape from -1 (concave) to +1 (convex)"
)
