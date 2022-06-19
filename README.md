# JJ.jl

Implementation of (some) [J](https://www.jsoftware.com) operators in Julia.

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

JJ.ranked(tuple, 0)(arr)  # Pack each element (rank 0 scalar) ... same as tuple.(arr)
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

## Further operators

As another example consider matrix multiplication.  Together with the
J-like operators `insert` and `table` matrix multiplication can be
expressed in several equivalent ways:

```julia
import JJ

A = JJ.iota(2, 3)
B = JJ.iota(3, 4)

@show A * B  # standard matrix multiplication

# In J notation: +/"1 A *"1/ |: B
@show JJ.ranked(x -> JJ.insert(+, x), 1)(JJ.table(JJ.ranked(1, JJ.ranked(0, *, 0), 1), A, B'))

# In J notation: +/ (|: A) */"1 B
@show JJ.insert(+, JJ.ranked(1, (x,y) -> JJ.table(JJ.ranked(0, *, 0), x, y), 1)(A', B))
```

Obviously not as concise as in J, but enough to illustrate the power
of these operators. For further examples see also my
[blog](https://bertschi.github.io/thinkapl.html) post.

Further inspiration:

* [JuliennedArrays.jl](https://github.com/bramtayl/JuliennedArrays.jl)

  Wonderful library to split-apply-combine n-dimensional arrays. Used
  here to implement rank functionality.
  
* [Rank in a hurry](https://code.jsoftware.com/wiki/Vocabulary/EZRank)

  Quick introduction to rank in J. Includes many additional examples
  and links.

