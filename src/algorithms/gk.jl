"""
Greenwald-Khanna (GK) Algorithm (Module GK)

Reference:
Greenwald, M., & Khanna, S. (2001). Space-efficient online computation of
quantile summaries. ACM SIGMOD Record, 30(2), 58-66.
"""

module GK
export GKEstimator, gk_init, gk_update!, gk_get_quantile

"""
    GKEstimator

Greenwald-Khanna epsilon-approx quantile summary.
- ε controls accuracy: error in rank <= εN
"""
struct GKEstimator
    ε::Float64
    n::Int
    summary::Vector{Tuple{Float64,Int,Int}}  # Each entry: (value, g, Δ)
end

function gk_init(ε::Float64)
    return GKEstimator(ε, 0, Vector{Tuple{Float64,Int,Int}}())
end

function gk_compress!(est::GKEstimator)
    # Compress the summary by merging buckets where possible
    summary = est.summary
    new_summary = Vector{Tuple{Float64,Int,Int}}()
    push!(new_summary, summary[1])
    for i in 2:length(summary)
        v,g,Δ = summary[i]
        v_prev, g_prev, Δ_prev = new_summary[end]
        # Check if can be merged
        if g_prev + g + Δ ≤ floor(2*est.ε*est.n)
            # Merge
            new_summary[end] = (v, g_prev+g, Δ_prev)
        else
            push!(new_summary, summary[i])
        end
    end
    return GKEstimator(est.ε, est.n, new_summary)
end

function gk_update!(est::GKEstimator, x::Float64)
    # Insert x into the summary
    n = est.n + 1
    s = est.summary

    # Use searchsortedfirst to get a single position
    pos = searchsortedfirst(s, (x, -Inf, -Inf), by=v->v[1])

    if isempty(s)
        # Empty summary: just insert
        s = [(x,1,0)]
    elseif pos == 1
        # Insert at the front
        insert!(s, 1, (x,1,0))
    elseif pos > length(s)
        # x is greater than all existing values, insert at the end
        push!(s, (x,1,0))
    else
        # Insert in the middle
        # Here we use the GK formula for Δ: floor(2 * ε * n)
        insert!(s, pos, (x,1,floor(2*est.ε*n)))
    end

    # Create a new estimator with the updated summary
    est = GKEstimator(est.ε, n, s)

    # Compress if needed
    if length(est.summary) > 1
        est = gk_compress!(est)
    end

    return est
end


function gk_get_quantile(est::GKEstimator, φ::Float64)
    # Find element q s.t. rank(q) ≈ φ * n
    r = φ * est.n
    rankmin = 0
    for (v,g,Δ) in est.summary
        rankmin += g
        if rankmin + Δ ≥ r
            return v
        end
    end
    return est.summary[end][1]
end

end # module GK