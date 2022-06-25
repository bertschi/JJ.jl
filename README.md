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

and a function providing some information about a value:
```julia
info(x) = "Some scalar $x"
info(x::AbstractArray) = "Array of shape $(size(x))"
```

Then, we can obtain information about each (scalar) element of an
array via broadcasting, i.e.,

```julia
info.(arr)
```

Alternatively, we can *rank* the `info` function:

```julia
using JJ

rank"info 0"(arr)  # Info on each element (rank 0 scalar) ... same as info.(arr)
rank"info 1"(arr)  # Info on each sub-vector (of rank 1)
rank"info 2"(arr)  # Info on each sub-matrix (of rank 2)
rank"info 3"(arr)  # Info on the whole array (of rank 3)
```

Note that

* sub-arrays are formed starting from the first dimensions[^noJ]

[^noJ]: This is different from J, but more suitable for arrays stored
    in column-major order. For easily comparing results with J the
    function `reversedims` can be used.

* the function is applied to each sub-array and must return results of
  the same type. These are then automatically combined into an array[^SAC].

[^SAC]: Ranking functions thereby provides a constraint version of the
    more general *split-apply-combine* strategy. In my opinion, the
    restricted, but very consistent model of J applying ranked
    functions on n-dimensional arrays is well designed and allows for
    surprisingly powerful and understandable code.

Thereby, many array manipulations can be expressed in a conscice
manner. In particular, *autobatching* ML-models is easily
accomplished:

```julia
using Flux

A = randn(2, 3, 5)  # a batch of 2x3 matrices
B = randn(3, 4, 5)  # another batch

A ⊠ B  # special batched matrix multiplication

rank"2 * 2"(A, B)  # just rank the standard one!

# Obviously this also works for complete models
# and across multiple sets of batches:
model = Dense(2, 4)
rank"model 1"(A)
```

Following J, ranking currently works for functions with one or two
arguments only. In case of two arguments, the function is broadcasted
across the left and right argument in order to be applied to all pairs
of sub-arrays. Here, in contrast to standard julia broadcasting,
missing dimensions are filled from the front in order to be consistent
with sub-arrays being formed from the front, i.e.,

```julia
A = randn(2, 3, 5)
B = randn(2, 3)
C = randn(2, 5)

dot(x, y) = sum(x .* y)  # function we want to use at rank 1 must work on vectors

rank"1 dot 1"(A, A)  # obviously works
rank"1 dot 1"(A, B)  # does not work as 3x5 does not match 3
rank"1 dot 1"(A, C)  # does work as 5 can be broadcasted over 3x5
```

## Further operators

As another example consider matrix multiplication.  Together with the
J-like operators `insert` and `table` matrix multiplication can be
expressed in several equivalent ways[^matmul]:

[^matmul]: While these are mathematically equivalent, direct matrix
    multiplication is implemented more efficiently.

```julia
using JJ

A = reshape(1:6, 2, 3)
B = reshape(1:12, 3, 4)

@show A * B  # standard matrix multiplication
# In J notation: A +/ . * B

@show rank"x -> insert(+, x) 1"(table(rank"1 rank\"0 * 0\" 1"(A', B)))
# In J notation: +/"1 A *"1/ |: B

# or slightly simpler using partial application and broadcasting instead of rank 0 function
partial(f, args...) = (moreargs...) -> f(args..., moreargs...)
@show rank"partial(insert, +) 1"(table(rank"1 .* 1", A', B))

@show insert(+, rank"1 partial(table, .*) 1"(A, B'))
# In J notation: +/ (|: A) */"1 B
```

Obviously not as concise as in J, but enough to illustrate the power
of these operators. For further examples, e.g., K-Means and
Transformer layers in J, see also my
[blog](https://bertschi.github.io/thinkapl.html) post[^JJ] or the
examples directory.

Further inspiration:

* [JuliennedArrays.jl](https://github.com/bramtayl/JuliennedArrays.jl)

  Wonderful library to split-apply-combine n-dimensional arrays. Used
  here to implement rank functionality.
  
* [Rank in a hurry](https://code.jsoftware.com/wiki/Vocabulary/EZRank)

  Quick introduction to rank in J. Includes many additional examples
  and links.
