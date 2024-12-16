import SequentialQuantileEstimator: P2
using CairoMakie
using Random
using Statistics

## Data (vector)
n = 3
data_sample = randn(n,1000)
true_med = median(data_sample, dims=2)
println("True medians: ", true_med)

## Suppose we want to track the median (0.5 quantile) for a vector of length 3:
est_vec = P2.p2_init([0.5], n)  # 3 scalar estimators, each tracking median
for i in axes(data_sample,2)
    P2.p2_update!(est_vec, vec(data_sample[:,i]))
end
med_vec = P2.p2_get_median(est_vec)  # returns a vector of 3 medians
println("Estimated medians: ", med_vec)