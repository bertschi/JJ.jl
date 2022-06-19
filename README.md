# JJ.jl

Implementation of (some) J operators in Julia.

## Verb rank

The J programming language has the interesting and useful concept of
*verb rank*. Thereby an operation can be applied at a specified depth
within a larger array.

Consider an `2 3 4` sized array 

```julia
arr = reshape(1:24, 2, 3, 4)
```

and a function packing an element into a `tuple`. Then, we can pack
each element via broadcasting, i.e.,

```julia
tuple.(arr)
```

Alternatively, we can *rank* the `tuple` function:

```julia
import JJ

JJ.ranked(tuple, 0)(arr)  # Pack each element (rank 0 scalar)
JJ.ranked(tuple, 1)(arr)  # Pack each sub-vector (of rank 1)
JJ.ranked(tuple, 2)(arr)  # Pack each sub-matrix (of rank 2)
JJ.ranked(tuple, 3)(arr)  # Pack the whole array (of rank 3)
```

Thereby, many array manipulations can be expressed in a conscice
manner. In particular, *autobatching* ML-models is easily
accomplished:

```julia
# Autobatch example from Jax (https://jax.readthedocs.io/en/latest/notebooks/vmapped_log_probs.html)

using Random
using Distributions

import JJ

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

batch_size = 12
batched_test_beta = rand(Normal(), batch_size, num_features)

# autobatch by applying vector function at rank 1
@show JJ.ranked(log_joint, 1)(batched_test_beta)
```
