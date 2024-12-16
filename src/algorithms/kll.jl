"""
KLL (Karnin, Lang, Liberty, 2016) Sketch (Module KLL)

Note: This is a simplified version of the KLL sketch algorithm.

Reference:
Karnin, Z., Lang, K., & Liberty, E. (2016, October). Optimal quantile
approximation in streams. In 2016 ieee 57th annual symposium on foundations of
computer science (focs) (pp. 71-78). IEEE.
"""

# module KLL
# export KLLEstimator, kll_init, kll_update!, kll_get_quantile

# """
#     KLLEstimator

# A simplified KLL sketch:
# - Keeps a set of levels, where each level is a reservoir of samples.
# - When full, levels are compressed by random sampling.
# """
# struct KLLEstimator
#     capacity::Int
#     data::Vector{Float64}
#     n::Int
# end

# function kll_init(capacity::Int=200)
#     return KLLEstimator(capacity, Float64[], 0)
# end

# function kll_update!(est::KLLEstimator, x::Float64)
#     data = est.data
#     push!(data, x)
#     n = est.n + 1
#     """
#     This deterministic "take every second element" approach is not the actual
#     KLL algorithm's method of compaction. The original KLL sketch uses a more
#     sophisticated, probabilistic compaction strategy to achieve its theoretical
#     error guarantees.
#     """
#     if length(data) > est.capacity
#         # compress by sampling half of the sorted data
#         sort!(data)
#         new_data = data[1:2:end] # take every second element
#         data = new_data
#     end
#     return KLLEstimator(est.capacity, data, n)
# end

# function kll_get_quantile(est::KLLEstimator, φ::Float64)
#     data = est.data
#     if isempty(data)
#         return NaN
#     end
#     sort!(data)
#     idx = clamp(ceil(Int, φ * length(data)), 1, length(data))
#     return data[idx]
# end

# end # module KLL

module KLL

export KLLEstimator, kll_init, kll_update!, kll_get_quantile

import Random

"""
    KLLEstimator

A structure for a KLL sketch.

Fields:
- capacity::Int: Maximum number of items per level before compaction.
- levels::Vector{Vector{Float64}}: An array of levels, each a vector of samples.
- total_count::Int: The total number of elements seen so far.

For simplicity, we fix the number of levels based on a capacity and log factor. In practice, you can dynamically add levels as needed.
"""
struct KLLEstimator
    capacity::Int
    levels::Vector{Vector{Float64}}
    total_count::Int
end

"""
    kll_init(capacity::Int=200)

Initialize a KLL estimator with a given per-level capacity.
"""
function kll_init(capacity::Int=200)
    # Start with a certain number of levels. The number of levels can be about O(log n),
    # but we will just allocate a small number and add as needed.
    levels = [Float64[] for _ in 1:8]  # 8 levels by default; you can adjust
    return KLLEstimator(capacity, levels, 0)
end

# Internal function: tries to compact a given level i into level i+1
function compact!(est::KLLEstimator, i::Int)
    lvl = est.levels[i]
    n = length(lvl)
    if n <= est.capacity
        return est
    end

    # Sort level before compaction
    sort!(lvl)
    # We'll keep half of them (rounded down)
    # Pair the elements: (lvl[1], lvl[2]), (lvl[3], lvl[4]), ...
    # For each pair, randomly choose one to keep.
    new_size = n ÷ 2
    new_samples = Vector{Float64}(undef, new_size)

    rng = Random.default_rng()
    for j in 1:new_size
        # pair: (lvl[2j-1], lvl[2j])
        # choose randomly one of them
        if rand(rng, Bool)
            new_samples[j] = lvl[2j-1]
        else
            new_samples[j] = lvl[2j]
        end
    end

    # Clear level i
    empty!(est.levels[i])

    # Add these sampled items to level i+1
    append!(est.levels[i+1], new_samples)

    # Potentially recursively compact i+1 if needed
    if length(est.levels[i+1]) > est.capacity
        if i+1 == length(est.levels)
            # If we are at the top and need more levels, create a new one
            push!(est.levels, Float64[])
        end
        est = compact!(est, i+1)
    end

    return est
end

"""
    kll_update!(est::KLLEstimator, x::Float64)

Insert a new value x into the KLL sketch. If level 0 overflows, it is compacted (randomly downsampled).
"""
function kll_update!(est::KLLEstimator, x::Float64)
    push!(est.levels[1], x)
    est = compact!(est, 1)
    return KLLEstimator(est.capacity, est.levels, est.total_count + 1)
end

"""
    kll_get_quantile(est::KLLEstimator, φ::Float64)

Return an approximate quantile for φ (0<φ<1).
We combine all levels into a single sorted list to answer queries.
This is a simple approach; more efficient methods can be used.
"""
function kll_get_quantile(est::KLLEstimator, φ::Float64)
    # Combine all samples
    all_samples = Float64[]
    for lvl in est.levels
        append!(all_samples, lvl)
    end
    if isempty(all_samples)
        return NaN
    end
    sort!(all_samples)
    idx = clamp(Int(ceil(φ * length(all_samples))), 1, length(all_samples))
    return all_samples[idx]
end

end # module KLL
