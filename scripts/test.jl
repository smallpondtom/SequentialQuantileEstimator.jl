import SequentialQuantileEstimator as SQE
using CairoMakie
using Distributions
using Random
using Statistics

## Generate the data samples (comment/uncomment the desired distribution)
Random.seed!(123) # Setting the seed
dist = Normal(0.1, 0.05)                                 # Normal distribution
# dist = MixtureModel([dist, Normal(0.5, 0.1)], [0.5, 0.5]) # Mixture of two normals
# dist = Beta(2.0, 5.0)                                     # Beta distribution
# dist = Exponential(0.5)                                   # Exponential distribution
# dist = Uniform(-1.0, 1.0)                                 # Uniform distribution
# dist = LogNormal(0.0, 0.5)                                # Log-normal distribution
# dist = Gamma(2.0, 0.5)                                    # Gamma distribution
# dist = Weibull(1.0, 0.5)                                  # Weibull distribution
# dist = Pareto(1.0, 0.5)                                   # Pareto distribution
# dist = Cauchy(0.0, 0.5)                                   # Cauchy distribution
# dist = Laplace(1.0, 0.5)                                  # Laplace distribution
data = rand(dist, 1000)
true_median = median(data)
println("True median: ", true_median)

## Initialize the estimators
meds = Dict(
    "P2" => Float64[],
    "GK" => Float64[],
    "t-Digest" => Float64[],
    "KLL" => Float64[]
)

## P² Example
est_p2 = SQE.P2.p2_init([0.5]) # median
for x in data
    est_p2 = SQE.P2.p2_update!(est_p2, x)
    push!(meds["P2"], SQE.P2.p2_get_quantiles(est_p2)[1])
end
println("P2 median estimate: ", SQE.P2.p2_get_quantiles(est_p2))

## GK Example
est_gk = SQE.GK.gk_init(0.01)
for x in data
    est_gk = SQE.GK.gk_update!(est_gk, x)
    push!(meds["GK"], SQE.GK.gk_get_quantile(est_gk, 0.5))
end
println("GK median estimate: ", SQE.GK.gk_get_quantile(est_gk, 0.5))

## t-Digest Example
est_t = SQE.TDigest.tdigest_init(100.0)
for x in data
    est_t = SQE.TDigest.tdigest_update!(est_t, x)
    push!(meds["t-Digest"], SQE.TDigest.tdigest_get_quantile(est_t, 0.5))
end
println("t-Digest median estimate: ", SQE.TDigest.tdigest_get_quantile(est_t, 0.5))

## KLL Example
est_kll = SQE.KLL.kll_init(200)
for x in data
    est_kll = SQE.KLL.kll_update!(est_kll, x)
    push!(meds["KLL"], SQE.KLL.kll_get_quantile(est_kll, 0.5))
end
println("KLL median estimate: ", SQE.KLL.kll_get_quantile(est_kll, 0.5))

## Plot the 
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1100, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Sample", ylabel = "Median Estimate", title=L"Median Estimation Per Sample for $\mathcal{N}(0.1, 0.05)$",
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(-20, length(data)+20, nothing, nothing)
    )
    l1 = hlines!(ax, true_median, color=:black, linewidth=4, linestyle=:dash)
    l2 = lines!(ax, 1:length(data), meds["P2"], linewidth=3)
    l3 = lines!(ax, 1:length(data), meds["GK"], linewidth=3)
    l4 = lines!(ax, 1:length(data), meds["t-Digest"], linewidth=3)
    l5 = lines!(ax, 1:length(data), meds["KLL"], linewidth=3)
    Legend(fig[1,2], 
        [l1,l2,l3,l4,l5], ["True Median", L"P^2", "GK", "t-Digest", "KLL"],
        labelsize=30
    )
    display(fig)
end