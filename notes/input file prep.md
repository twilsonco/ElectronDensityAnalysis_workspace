# TOML input file support for ElectronDensityAnalysis.jl

I'm developing a Julia package called ElectronDensityAnalysis.jl. It's added as a dependency in this directory. The package is meant to be used not only in writing Julia programs to conduct electron density analysis, but also to be run standalone to run analysis of electron charge density data, producing output files. For this latter function, I need to add input file a.k.a. run file support. Run files will be TOML files that can be natively read by Julia, and will specify things like:

- The input file path (later with the ability to specify multiple input files, or to specify an input directory)
- The output directory
- The input file(s) type (Tape21 | Tape41-ADF | Tape41-BAND | CHGCAR | CUBE-Gaussian | CUBE-FlapW | CUBE-QE | CUBE-TURBOMOLE) to control which function is used to load the data
- Grid operations: expand translationally periodic system or mirror system with mirror symmetry
- Perform analysis:
  - Find critical points
  - Perform gradient bundle decomposition (a.k.a. gradient bundle condensation)
  - Identify gradient bundle condensed basins

Now, most of these things will have a [table] in the TOML input file to have the feel of a section header.
