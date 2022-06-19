
# Autobatch example from Jax (https://jax.readthedocs.io/en/latest/notebooks/vmapped_log_probs.html)

using Random
using Distributions
using Flux

include("JJ.jl")

# generate fake binary classification dataset
Random.seed!(10009)

num_features = 10
num_points = 100

true_beta = rand(Normal(), num_features)
all_x = rand(Normal(), num_points, num_features)
y = rand(Normal(), num_points) .< Flux.sigmoid.(all_x * true_beta)


function log_joint(beta::AbstractVector)
    # Note that no `dims` parameter is provided to `sum`.
    sum(logpdf.(Normal(0, 1), beta)) + sum(.- log.(1 .+ exp.(.- (2 .* y .- 1) .* (all_x * beta))))
end

function batched_log_joint(beta::AbstractArray{T,2}) where {T}
    # Here (and below) `sum` needs an `dims` parameter. At best, forgetting to set axis
    # or setting it incorrectly yields an error; at worst, it silently changes the
    # semantics of the model.
    logprior = sum(logpdf.(Normal(0, 1), beta); dims=2)
    # Note the multiple transposes. Getting this right is not rocket science,
    # but it's also not totally mindless. (I didn't get it right on the first
    # try.)
    # Note the differences in broadcasting between Python (from end) and Julia (from beginning)
    loglik = sum(.- log.(1 .+ exp.(.- (2 .* y[[CartesianIndex()],:] .- 1) .* (all_x * beta')')); dims=2)
    return logprior + loglik
end

batch_size = 10
batched_test_beta = rand(Normal(), batch_size, num_features)

@show batched_log_joint(batched_test_beta)
# autobatch by applying vector function at rank 1
@show JJ.ranked(log_joint, 1)(batched_test_beta)

