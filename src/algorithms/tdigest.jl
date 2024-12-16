"""
t-Digest (Module TDigest)

Note: This implementation is a simplified version of the t-digest algorithm.

References:
[1] Dunning, T., & Ertl, O. (2019). Computing extremely accurate quantiles using
t-digests. arXiv preprint arXiv:1902.04023.
[2] Dunning, T. (2021). The t-digest: Efficient estimates of distributions.
Software Impacts, 7, 100049.
"""

module TDigest
export TDigestEstimator, tdigest_init, tdigest_update!, tdigest_get_quantile

"""
    TDigestEstimator

A simplified t-digest structure.
- Uses a list of centroids (mean, weight).
- δ controls how tightly centroids are packed around the median.
"""
struct TDigestEstimator
    δ::Float64
    centroids::Vector{Tuple{Float64,Float64}}
    n::Float64
end

function tdigest_init(δ::Float64=100.0)
    return TDigestEstimator(δ, Float64[], 0.0)
end

function tdigest_update!(est::TDigestEstimator, x::Float64)
    # Insert x by merging into closest centroid or creating a new one
    if isempty(est.centroids)
        est = TDigestEstimator(est.δ, [(x,1.0)], 1.0)
        return est
    end

    # Extract means from the centroids
    means = first.(est.centroids)  # vector of means
    # Find closest centroid to x
    c_idx = argmin(abs.(means .- x))
    m,w = est.centroids[c_idx]

    # k-size function (currently not used, but fixed to avoid error):
    k = (est.n * (means .- x) ./ est.n)

    # Merge into that centroid
    new_m = (m*w + x)/(w+1.0)
    new_w = w + 1.0
    new_centroids = copy(est.centroids)
    new_centroids[c_idx] = (new_m, new_w)
    est = TDigestEstimator(est.δ, new_centroids, est.n+1.0)
    return est
end

function tdigest_cdf(est::TDigestEstimator, x::Float64)
    # Approximate CDF by walking through centroids
    # For simplicity: linear interpolation between centroids
    if isempty(est.centroids)
        return 0.0
    end
    sorted_c = sort(est.centroids, by=c->c[1])
    total = est.n
    cum = 0.0
    for i in eachindex(sorted_c)
        m,w = sorted_c[i]
        if x < m
            # linear interpolation with previous
            if i == 1
                return (x < m) ? 0.0 : w/total
            else
                m_prev,w_prev = sorted_c[i-1]
                # interpolate
                ratio = (x - m_prev)/(m - m_prev)
                return (cum + ratio*w_prev)/total
            end
        end
        cum += w
    end
    return 1.0
end

function tdigest_get_quantile(est::TDigestEstimator, φ::Float64)
    # Inverse CDF: binary search for value x s.t. tdigest_cdf(x) ≈ φ
    # Simplified: we'll just do a linear search (not efficient)
    if isempty(est.centroids)
        return NaN
    end
    sorted_c = sort(est.centroids, by=c->c[1])
    target = φ * est.n
    cum = 0.0
    for i in eachindex(sorted_c)
        m,w = sorted_c[i]
        if cum + w ≥ target
            # interpolate
            if w == 1.0 || i == 1
                return m
            else
                # linear interpolation
                prev_m, prev_w = (i > 1) ? sorted_c[i-1] : (m,w)
                local_ratio = (target - cum)/w
                return prev_m + (m - prev_m)*local_ratio
            end
        end
        cum += w
    end
    return sorted_c[end][1]
end

end # module TDigest