"""
P² Algorithm Implementation (Module P2)

Reference:
Jain, R., & Chlamtac, I. (1985). The P2 algorithm for dynamic calculation of
quantiles and histograms without storing observations. Communications of the
ACM, 28(10), 1076-1085.
"""

module P2
export P2Estimator, p2_init, p2_update!, p2_get_quantiles

"""
    P2Estimator

Holds state for the P² quantile estimation algorithm.
"""
struct P2Estimator
    n::Int
    q::Vector{Float64}
    n_markers::Vector{Int}
    n_desired::Vector{Float64}
    quantiles::Vector{Float64}
    initialized::Bool
end

"""
    p2_init(quantiles::Vector{Float64})

Initialize a P² estimator for given quantiles (excluding 0 and 1, these are automatically included).
"""
function p2_init(quantiles::Vector{Float64})
    @assert all(0 .< quantiles .< 1) "Quantiles must be in (0,1)."
    @assert issorted(quantiles)
    return P2Estimator(0, Float64[], Int[], Float64[], quantiles, false)
end

function p2_initialize!(est::P2Estimator, data::Vector{Float64})
    sort!(data)
    full_q = vcat(0.0, est.quantiles, 1.0)
    m = length(full_q)
    N = length(data)
    q_positions = 1 .+ full_q .* (N - 1)
    q_pos_rounded = round.(Int, q_positions)

    q = similar(data, m)
    for i in 1:m
        q[i] = data[q_pos_rounded[i]]
    end

    n_markers = q_pos_rounded
    n_desired = q_positions

    return P2Estimator(N, q, n_markers, n_desired, est.quantiles, true)
end

function p2_update!(est::P2Estimator, x::Float64)
    m = length(est.quantiles) + 2

    if !est.initialized
        buffer = vcat(est.q, [x])
        if length(buffer) < m
            return P2Estimator(est.n+1, buffer, est.n_markers, est.n_desired, est.quantiles, false)
        else
            return p2_initialize!(est, buffer)
        end
    end

    # Already initialized
    n = est.n + 1
    q = copy(est.q)
    n_markers = copy(est.n_markers)
    n_desired = copy(est.n_desired)
    full_q = vcat(0.0, est.quantiles, 1.0)

    # Insert x
    if x < q[1]
        q[1] = x
        pos = 1
    elseif x >= q[end]
        q[end] = x
        pos = m-1
    else
        pos = 1
        for i in 1:(m-1)
            if x < q[i+1]
                pos = i
                break
            end
        end
    end

    for j in pos+1:m
        n_markers[j] += 1
    end

    for i in 1:m
        n_desired[i] += full_q[i]
    end

    # Adjust markers
    for i in 2:(m-1)
        d = n_desired[i] - n_markers[i]
        if (d ≥ 1 && n_markers[i+1] - n_markers[i] > 1) || (d ≤ -1 && n_markers[i-1] - n_markers[i] < -1)
            d_sign = sign(d)
            q_new = q[i] + d_sign * (
                ((n_markers[i] - n_markers[i-1] + d_sign)*(q[i+1] - q[i])/(n_markers[i+1] - n_markers[i]) +
                 (n_markers[i+1] - n_markers[i] - d_sign)*(q[i] - q[i-1])/(n_markers[i] - n_markers[i-1])
                )/(n_markers[i+1] - n_markers[i-1])
            )
            q_new = clamp(q_new, min(q[i-1], q[i+1]), max(q[i-1], q[i+1]))
            q[i] = q_new
            n_markers[i] += d_sign
        end
    end

    return P2Estimator(n, q, n_markers, n_desired, est.quantiles, true)
end

function p2_get_quantiles(est::P2Estimator)
    if !est.initialized
        data = est.q
        sort!(data)
        return [data[ceil(Int, q*length(data))] for q in est.quantiles]
    else
        # q: [0, q1, q2,..., 1]
        # quantiles are from index 2 to m-1
        m = length(est.quantiles) + 2
        return [est.q[i] for i in 2:(m-1)]
    end
end

#===============================================#
"""
Function dispatches for vector and matrix data.
"""
#===============================================#

# For convenience, define a helper initializer that creates one estimator per element
function p2_init_for_vector(quantiles::Vector{Float64}, length::Int)
    return [p2_init(quantiles) for _ in 1:length]
end

function p2_init_for_matrix(quantiles::Vector{Float64}, rows::Int, cols::Int)
    return [p2_init(quantiles) for _ in 1:rows, _ in 1:cols]
end

"""
    p2_init(quantiles::Vector{Float64}, data::Vector{Float64})

Initialize a P² estimator array for vector data. The returned object is a vector of `P2Estimator`s, one per element.
If `data` is provided, it will be used to initialize the estimators directly.
"""
function p2_init(quantiles::Vector{Float64}, data::Vector{Float64})
    est_array = p2_init_for_vector(quantiles, length(data))
    # Use the initial data to "warm up" each estimator
    for i in 1:length(data)
        # For the scalar estimator, we need at least (length(quantiles)+2) points to init properly.
        # So we just feed this single initial data point and rely on streaming updates. 
        # If you have multiple initial data vectors you could feed them in here.
        est_array[i] = p2_update!(est_array[i], data[i])
    end
    return est_array
end

"""
    p2_init(quantiles::Vector{Float64}, length::Int)

Initialize a P² estimator array for vector data of given length, but without initial data.
"""
function p2_init(quantiles::Vector{Float64}, length::Int)
    return p2_init_for_vector(quantiles, length)
end

"""
    p2_init(quantiles::Vector{Float64}, data::Matrix{Float64})

Initialize a P² estimator array for matrix data. The returned object is an MxN array of `P2Estimator`s.
If `data` is provided, it will be used to initialize the estimators directly.
"""
function p2_init(quantiles::Vector{Float64}, data::Matrix{Float64})
    (rows, cols) = size(data)
    est_array = p2_init_for_matrix(quantiles, rows, cols)
    # Initialize each estimator with the corresponding element
    for i in 1:rows
        for j in 1:cols
            est_array[i,j] = p2_update!(est_array[i,j], data[i,j])
        end
    end
    return est_array
end

"""
    p2_init(quantiles::Vector{Float64}, rows::Int, cols::Int)

Initialize a P² estimator array for matrix data with given dimensions, but no initial data.
"""
function p2_init(quantiles::Vector{Float64}, rows::Int, cols::Int)
    return p2_init_for_matrix(quantiles, rows, cols)
end

"""
    p2_update!(est_array::Vector{P2Estimator}, x::Vector{Float64})

Update a vector of P² estimators with a new vector of data.
Each element's estimator is updated with the corresponding element of `x`.
"""
function p2_update!(est_array::Vector{P2Estimator}, x::Vector{Float64})
    @assert length(est_array) == length(x) "Length mismatch between estimator array and data vector."
    for i in 1:length(x)
        est_array[i] = p2_update!(est_array[i], x[i])
    end
    return est_array
end

"""
    p2_update!(est_array::Matrix{P2Estimator}, X::Matrix{Float64})

Update an MxN array of P² estimators with a new MxN matrix of data.
Each estimator is updated with the corresponding element of `X`.
"""
function p2_update!(est_array::Matrix{P2Estimator}, X::Matrix{Float64})
    @assert size(est_array) == size(X) "Size mismatch between estimator array and data matrix."
    (rows, cols) = size(X)
    for i in 1:rows
        for j in 1:cols
            est_array[i,j] = p2_update!(est_array[i,j], X[i,j])
        end
    end
    return est_array
end

"""
    p2_get_median(est_array::Vector{P2Estimator})

Compute the median for each element of the vector estimators.
Returns a vector of median values of the same length.
"""
function p2_get_median(est_array::Vector{P2Estimator})
    # The median is quantile 0.5
    return [p2_get_quantiles(e)[1] for e in est_array]
end

"""
    p2_get_median(est_array::Matrix{P2Estimator})

Compute the median for each element of the matrix estimators.
Returns an MxN matrix of median values.
"""
function p2_get_median(est_array::Matrix{P2Estimator})
    (rows, cols) = size(est_array)
    medians = Matrix{Float64}(undef, rows, cols)
    for i in 1:rows
        for j in 1:cols
            medians[i,j] = p2_get_quantiles(est_array[i,j])[1]
        end
    end
    return medians
end

end # module P2