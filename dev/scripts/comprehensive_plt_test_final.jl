using ElectronDensityAnalysis
using LinearAlgebra
include("../test/test_helpers.jl")

function test_comprehensive_plt_writer()
    println("=== Comprehensive TecplotWriter Test ===")
    println("Testing both synthetic and real chemical systems")

    # Test 1: Synthetic system (quick test)
    println("\n--- Test 1: Synthetic System ---")

    # Create a simple synthetic system like we had before
    origin = [0.0, 0.0, 0.0]
    lattice = [0.1 0.0 0.0; 0.0 0.1 0.0; 0.0 0.0 0.1]
    n_pts = [10, 10, 10]
    gs = GridSpec(origin, lattice, n_pts)

    atoms = [
        NuclearCoordinate([0.3, 0.4, 0.5], 6),   # Carbon - within grid bounds
        NuclearCoordinate([0.6, 0.4, 0.5], 1),   # Hydrogen - within grid bounds
    ]

    data = zeros(Float32, n_pts...)
    for i in 1:n_pts[1], j in 1:n_pts[2], k in 1:n_pts[3]
        pos = [i-1, j-1, k-1] .* 0.1
        for atom in atoms
            r = norm(pos - atom.r)
            amplitude = atom.data.number == 6 ? 2.0 : 0.5
            data[i, j, k] += amplitude * exp(-r^2 / 0.2)
        end
    end

    # Create interpolation functions
    g = generate_interpolation_grid(gs)
    itp = interpolate((g[1], g[2], g[3]), data, Gridded(Linear()))

    func_xyz = (r) -> itp(r...)
    func_single = (r) -> func_xyz(r)
    grad_func = (r) -> [0.0, 0.0, 0.0]  # Dummy
    hess_func = (r) -> diagm([1.0, 1.0, 1.0])  # Dummy

    func! = (result, r) -> result[1] = func_single(r)
    grad! = (result, r) -> result .= grad_func(r)
    hess! = (result, r) -> result .= hess_func(r)

    synthetic_sys = System(
        "Synthetic Test System",
        "TecplotWriter Comprehensive Test",
        atoms,
        gs,
        data,
        [false, false, false],
        func_xyz,
        func_single,
        grad_func,
        hess_func,
        func!,
        grad!,
        hess!,
    )

    # Write synthetic system
    try
        TecplotWriter.write_plt_file(
            "comprehensive_test_synthetic.plt",
            Tuple{String,Any}[("Synthetic System", synthetic_sys)];
            title="Synthetic System Test",
            system=synthetic_sys,
        )
        println("✅ Synthetic system PLT file written successfully")
    catch e
        println("❌ Synthetic system test failed: $e")
    end

    # Test 2: Real chemical system
    println("\n--- Test 2: Real Chemical System ---")

    try
        # Load real system
        real_sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
        println("System: $(real_sys.name)")
        println("Grid: $(real_sys.grid.n_pts) points")
        println("Atoms: $(length(real_sys.atoms))")

        # Find critical points
        cps = find_critical_points(real_sys)
        println("Critical points found: $(length(cps))")

        # Group by type
        cp_type_counts = Dict{String,Int}()
        for cp in cps
            type_str = string(cp.type)
            cp_type_counts[type_str] = get(cp_type_counts, type_str, 0) + 1
        end

        println("CP types:")
        for (type_str, count) in cp_type_counts
            println("  $type_str: $count")
        end

        # Write comprehensive PLT file with everything
        zones = Tuple{String,Any}[("Real System", real_sys), ("Critical Points", cps)]

        TecplotWriter.write_plt_file(
            "comprehensive_test_real.plt",
            zones;
            title="Real Chemical System - $(real_sys.name)",
            system=real_sys,
        )

        println("✅ Real system PLT file written successfully")
        println("  Grid data: $(prod(real_sys.grid.n_pts)) points")
        println("  Atoms: $(length(real_sys.atoms)) atoms")
        println("  Critical points: $(length(cps)) CPs")

    catch e
        println("❌ Real system test failed: $e")
        showerror(stdout, e)
        println()
    end

    # Test 3: Mixed data types
    println("\n--- Test 3: Mixed Data Types ---")

    try
        # Use synthetic system but with more complex structures
        simple_cps = [
            CriticalPoint([0.3, 0.4, 0.5], nuclear_cp, 1),
            CriticalPoint([0.6, 0.4, 0.5], bond_cp, 0),
        ]

        mixed_zones = Tuple{String,Any}[
            ("Mixed System", synthetic_sys), ("Mixed CPs", simple_cps)
        ]

        TecplotWriter.write_plt_file(
            "comprehensive_test_mixed.plt",
            mixed_zones;
            title="Mixed Data Types Test",
            system=synthetic_sys,
        )

        println("✅ Mixed data types PLT file written successfully")

    catch e
        println("❌ Mixed data types test failed: $e")
        showerror(stdout, e)
        println()
    end

    # Test 4: Gradient Paths with Real System
    println("\n--- Test 4: Gradient Paths ---")

    try
        # Load real system for gradient path analysis
        gp_sys = with_zip_file("test_data/ethane_adf.t41", load_data_scm_adf_t41)[1]
        println("Creating gradient paths for: $(gp_sys.name)")

        # Find critical points first
        gp_cps = find_critical_points(gp_sys)
        println("Found $(length(gp_cps)) critical points for gradient path seeding")

        # Create several gradient paths from different starting points
        gradient_paths = GradientPath[]

        # Create paths from points near the molecular center
        seed_points = [
            [0.0, 0.0, 0.1],   # Near center
            [0.5, 0.0, 0.0],   # Offset in x
            [0.0, 0.5, 0.0],   # Offset in y
            [-0.5, 0.0, 0.0],  # Negative x offset
        ]

        for (i, seed_point) in enumerate(seed_points)
            try
                # Create gradient path with critical points context
                gp = create_gradient_path(seed_point, gp_sys, both_dir; cps=gp_cps)
                push!(gradient_paths, gp)
                println("  Created gradient path $i: $(size(gp.r, 2)) points")
            catch e
                println("  Warning: Could not create gradient path $i from $seed_point: $e")
            end
        end

        println("Successfully created $(length(gradient_paths)) gradient paths")

        # Write comprehensive PLT file with system, critical points, and gradient paths
        gp_zones = Tuple{String,Any}[("GP System", gp_sys), ("GP Critical Points", gp_cps)]

        # Add each gradient path as a separate zone
        for (i, gp) in enumerate(gradient_paths)
            push!(gp_zones, ("Gradient Path $i", gp))
        end

        TecplotWriter.write_plt_file(
            "comprehensive_test_gradpaths.plt",
            gp_zones;
            title="Gradient Paths Analysis - $(gp_sys.name)",
            system=gp_sys,
        )

        println("✅ Gradient paths PLT file written successfully")
        println("  System: $(gp_sys.name)")
        println("  Critical points: $(length(gp_cps)) CPs")
        println("  Gradient paths: $(length(gradient_paths)) paths")
        total_gp_points = sum(size(gp.r, 2) for gp in gradient_paths)
        println("  Total GP points: $total_gp_points")

    catch e
        println("❌ Gradient paths test failed: $e")
        showerror(stdout, e)
        println()
    end

    println("\n=== Comprehensive Test Summary ===")
    println("Generated files:")
    println("  📁 comprehensive_test_synthetic.plt - Simple synthetic system")
    println("  📁 comprehensive_test_real.plt - Real ethane system with CPs")
    println("  📁 comprehensive_test_mixed.plt - Mixed synthetic/CP data")
    println("\n🎉 Comprehensive TecplotWriter test completed!")
    println("All files are ready for visualization in Tecplot 360 or ParaView")
end

# Run the comprehensive test
test_comprehensive_plt_writer()
