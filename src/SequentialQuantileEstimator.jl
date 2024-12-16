module SequentialQuantileEstimator

# Include each algorithm's source file
include("algorithms/p2.jl")
include("algorithms/gk.jl")
include("algorithms/tdigest.jl")
include("algorithms/kll.jl")

# Make each submodule available through this top-level module
using .P2
using .GK
using .TDigest
using .KLL

end # module SequentialQuantileEstimator
