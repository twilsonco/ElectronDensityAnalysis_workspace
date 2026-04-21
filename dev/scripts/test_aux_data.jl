using ElectronDensityAnalysis
using LinearAlgebra
include("../test/test_helpers.jl")

function test_aux_data_feature()
    println("=== Testing Auxiliary Data Feature ===")

    # Create a simple system for testing
    origin = [0.0, 0.0, 0.0]
    lattice = [0.1 0.0 0.0; 0.0 0.1 0.0; 0.0 0.0 0.1]
    n_pts = [5, 5, 5]
    gs = GridSpec(origin, lattice, n_pts)

    atoms = [
        NuclearCoordinate([0.2, 0.2, 0.2], 6),   # Carbon
        NuclearCoordinate([0.3, 0.3, 0.3], 1),   # Hydrogen
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

    test_sys = System(
        "Test System with Aux Data",
        "Demonstration of auxiliary data feature",
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

    # Create some critical points for demonstration
    cps = [
        CriticalPoint([0.2, 0.2, 0.2], nuclear_cp, 1),
        CriticalPoint([0.25, 0.25, 0.25], bond_cp, 0),
    ]

    # Define auxiliary data that will be passed down hierarchically
    aux_data = [
        "Author=GitHub Copilot",
        "CreationDate=2025-07-31",
        "ProjectName=ElectronDensityAnalysis.jl",
        "Purpose=Demonstration of auxiliary data feature",
        "Version=1.0",
        "Department=Molecular Theory Group",
    ]

    # Write PLT file with aux_data
    zones = Tuple{String,Any}[("Test System", test_sys), ("Test Critical Points", cps)]

    TecplotWriter.write_plt_file(
        "test_aux_data.plt",
        zones;
        title="Auxiliary Data Demonstration",
        system=test_sys,
        aux_data=aux_data,  # This will be written to dataset level and passed to all zones
    )

    println("✅ PLT file with auxiliary data written successfully")
    println("Dataset-level aux data:")
    for aux in aux_data
        println("  - $aux")
    end

    println("\nZone-level aux data includes:")
    println("  - Dataset aux data (inherited)")
    println("  - Zone-specific aux data (e.g., ZoneType=System)")
    println("  - Child zone aux data (e.g., ParentZone=Test System, AtomType=C)")

    println("\n🎉 Auxiliary data feature demonstration completed!")
    println("Open 'test_aux_data.plt' in Tecplot to view the auxiliary data")
end

test_aux_data_feature()
