# # A gaussian mixture model
# First of all we define our model,
using KissABC
using Distributions

function model(P, N)
    μ_1, μ_2, σ_1, σ_2, prob = P
    d1 = randn(N) .* σ_1 .+ μ_1
    d2 = randn(N) .* σ_2 .+ μ_2
    ps = rand(N) .< prob
    R = zeros(N)
    R[ps] .= d1[ps]
    R[.!ps] .= d2[.!ps]
    R
end

# Let's use the model to generate some data, this data will constitute our dataset
parameters = (1.0, 0.0, 0.2, 2.0, 0.4)
data = model(parameters, 5000)

# let's look at the data

using Plots
histogram(data)
savefig("ex1_hist1.svg");
nothing; # hide

# ![ex1_hist1](ex1_hist1.svg)

# we can now try to infer all parameters using `KissABC`, first of all we need to define a reasonable prior for our model

prior = Factored(
    Uniform(0, 2), # there is surely a peak between 0 and 2
    Uniform(-1, 1), #there is a smeared distribution centered around 0
    Uniform(0, 1), # the peak has surely a width below 1
    Uniform(0, 4), # the smeared distribution surely has a width less than 4
    Beta(2, 2), # the number of total events from both distributions look about the same, so we will favor 0.5 just a bit
);

# let's look at a sample from the prior, to see that it works
rand(prior)
# now we need a function to compute summary statistics for our data, this is not the optimal choice, but it will work out anyway
function S(x)
    r = (0.1, 0.2, 0.45, 0.55, 0.8, 0.9)
    quantile(x, r)
end

# we will define a function to use the `model` and summarize it's results
summ_model(P, N) = S(model(P, N));

# now we need a distance function to compare the summary statistics of target data and simulated data
summ_data = S(data)
D(P, N = 5000) = sqrt(mean(abs2, summ_data .- summ_model(P, N)));

# We can use `AIS` which is an Affine Invariant MC algorithm via the `sample` function, to get the posterior distribution of our parameters given the dataset `data`
approx_density = ApproxPosterior(prior, D, 0.1)
res = sample(
    approx_density,
    AIS(50),
    MCMCThreads(),
    1000,
    4,
    burnin = 300,
    ntransitions = 10,
    progress = false,
)
@show res
# the nominal values of the parameters lie inside the CI.
