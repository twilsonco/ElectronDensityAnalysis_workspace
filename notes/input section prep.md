OK, that sounds fine. I'd like to use a different name than "MyAnalysisTool", how about "ElectronDenistyAnalysisRunner" in a file `electron_density_analysis_runner.jl` file instead?

That aside, I think we're ready to start implementing this. I have a workspace setup that loads `ElectronDensityAnalysis.jl` as a dependency, where we can write and test the runner functionality. I have an outer directory, inside which is `config/`, `runner.jl`, and `electron_density_analysis_runner.jl` (both blank).

The first section to start with will be the input section. This needs to be flexible, so that the user can specify

- input: a union that could be:
  - Path to one or more input file (wildcard string)
  - List of paths to input files (vector of wildcard strings)
  - Path to a directory of input files (wildcard string)
  - Specification of input file path(s) with associated names (vector of named tuples, e.g. `[(name::String, path::String), ...]`), which allows the user to specify custom names for each input file that can be used in the output file naming.
- type: an Enum specifying the input type ( Tape41-ADF | Tape41-BAND | CHGCAR | AECCAR | CUBE | CUBE-ADF | CUBE-Gaussian | CUBE-QE | CUBE-TURBOMOLE | CHARGE-FlapW | PLT-TURBOMOLE); applies to all input files specified in `input`
- [optional] periodic: a list of bools (or 1/0's) specifying whether the system is periodic in the x, y, and z directions; default depends on the `type` (vasp files—CHGCAR/AECCAR—are assumed periodic in all directions, otherwise default is non-periodic in all directions)
  - (During config struct construction, this will be used to create a `Vector{Bool}` of length 3 for the periodicity in each direction.)
- [optional] 64bit: a bool specifying whether to load the data in 64-bit precision (Float64) or 32-bit precision (Float32); default depends on `type` (text files (CUBE/CHGCAR/AECCAR/CHARGE) are assumed 32-bit, while binary files (the remainder) default to 64-bit)
  - (During config struct construction, this will be used to set a `Type` variable to `Float32` or `Float64`.)
- [optional] variables: for Tape41-ADF and Tape41-BAND only, a vector of strings specifying which variables to load from the file. Default is "SCF/Density" for ADF or "FOO/rho" for BAND files.
  - The first will be the "main" variable used for analysis, while the remainder will be additional scalar fields included during integration for gradient bundle decomposition, so that you can, e.g. integrate the potential and/or kinitic energy densities during gradient bundle decomposition. These are case-sensitive as "section/variable". To see what sections and variables are available in a file, the user can open the Tape41 file in "KF Browser" using AMS (or the older ADF) GUI, then enable "File -> Expert Mode", then the sections will appear at top-level sections, and when expanded, the next level will be the variable names. You must specify the section and variable name exactly as they appear in the KF Browser, including capitalization, to load the variable successfully. Alternatively, use the `$AMSBIN/dmpkf <path/to/tape41> --toc` to print the table of contents of the file, which will show the section and variable names. Remember that any Tape41 you generate will only contain the variables that are specified in the run file used to generate the Tape41.
