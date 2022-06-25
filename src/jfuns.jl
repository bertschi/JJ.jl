
"""
    reversedims(A)

Reverses dimensions of A. Convenient for comparing results with J.

# Examples

Corresponds to `i. 2 3` (iota) in J
```julia-repl
julia> reversedims(reshape(1:6, 3, 2))
2×3 Matrix{Int64}:
 1  2  3
 4  5  6
```
"""
reversedims(x::AbstractArray) = permutedims(x, ndims(x):-1:1)

"""
    table(fun, x, y)

Applies `fun` to `x` and `y` in a tabular fashion, i.e., like an outer product.
"""
function table end

_table(fun, x, y) = fun(x, y)

function _table(fun, x, y::AbstractArray)
    combine(broadcast(fun, x, y))
end

function _table(fun, x::AbstractArray, y)
    combine(broadcast(fun, x, y))
end

function _table(fun, x::AbstractArray, y::AbstractArray)
    xl = reshape(x, (size(x)..., ntuple(i -> 1, ndims(y))...)...)
    yr = reshape(y, (ntuple(i -> 1, ndims(x))..., size(y)...)...)
    combine(broadcast(fun, xl, yr))
end

"""
    table(fun, x, y)

Computes `fun(x[i], y[j])` for eachindex of `x` and `y` respectively
all of which must have the same size `size(fxy)`. All results are then
collect in an array of size `size(x)` times `size(y)` times `size(fxy)`.

Scalar arguments are considered as 0-dimensional arrays.

# Examples

Multiplication table
```julia-repl
julia> table(*, 1:3, 1:3)
3×3 Matrix{Int64}:
 1  2  3
 2  4  6
 3  6  9
```

Larger example with vector valued function
```julia-repl
julia> size(table((x,y) -> [x, y], ones(3, 4), zeros(5)))
(2, 3, 4, 5)
```
"""
function table(fun, x, y)
    _table(fun, x, y)
end

"""
    table(fd::RankedDyad, x, y)

Table respects the left and right rank of the provided function and
computes `fun(x[i], y[j])` only on eachindex of the corresponding
parts remaining of `x` and `y` after forming subarrays of the left and
right rank of `fd` respectively.

# Example

Matrix product (note that the left argument is transposed)
```julia-repl
julia> table(rank\"1 (x, y) -> sum(x .* y) 1\", reshape(1:6, 2, 3)', reshape(1:12, 3, 4))
2×4 Matrix{Int64}:
 22  49   76  103
 28  64  100  136
```
"""
table(fd::RankedDyad, x, y) = fd.fun(x, y)

function table(fd::RankedDyad{F,L,R}, x::AbstractArray, y::AbstractArray) where {F,L,R}
    _table(fd.fun, enframe(x, Val(L)), enframe(y, Val(R)))
end

function table(fd::RankedDyad{F,L,R}, x, y::AbstractArray) where {F,L,R}
    _table(fd.fun, x, enframe(y, Val(R)))
end

function table(fd::RankedDyad{F,L,R}, x::AbstractArray, y) where {F,L,R}
    _table(fd.fun, enframe(x, Val(L)), y)
end

"""
    insert(fun, x)

Inserts `fun` between all items, i.e, sub-arrays sliced along the last
dimension, of `x`. As in J, associates `fun` to the right.

# Examples

In contrast to `sum(+, [1 2 3; 4 5 6]; dims=2)` the
dimension of the resulting array is always reduced by one.
```julia-repl
julia> insert(+, [1 2 3; 4 5 6])
2-element Vector{Int64}:
  6
 15
```

Same as `1 - (2 - 3)`
```julia-repl
julia> insert(-, [1, 2, 3])
2
```
"""
function insert(fun, x::AbstractArray{T,N}; init=Base._InitialValue()) where {T,N}
    foldr(fun, enframe(x, Val(N-1)); init=init)
end
