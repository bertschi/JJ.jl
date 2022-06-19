
"""
    iota(dims...)

N-dimensional array with the specified dimensions containing integers from 0.
As in J, entries vary fastest along the last dimension.

Convenient for comparing functions against J.

# Examples
```julia-repl
julia> iota(2, 3)
2×3 Matrix{Int64}:
 1  2  3
 4  5  6
```
"""
function iota(dims...)
    permutedims(
        reshape((1:prod(dims)) .- 1, reverse(dims)),
        reverse(1:length(dims)))
end

"""
    table(fun, x, y)

Applies `fun` to `x` and `y` in a tabular fashion, i.e., like an outer product.
"""
function table end

"""
    table(fun, x, y)

Equivalent to `fun(x, y)` on scalar, i.e., non-array elements.
"""
table(fun, x, y) = fun(x, y)  # scalar default

"""
    table(fun, x::AbstractArray, y::AbstractArray)

Computes `fun(x[i], y[j])` for eachindex of `x` and `y` respectively
all of which must have the same size `size(fxy)`. All results are then
collect in an array of size `size(x)` times `size(y)` times `size(fxy)`.

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
julia> size(table((x,y) -> [x, y], ones(2, 3), zeros(4)))
(2, 3, 4, 2)
```
"""
function table(fun, x::AbstractArray, y::AbstractArray)
    combine(
        reshape([fun(x[i], y[j])
                 for i in eachindex(x), j in eachindex(y)],
                (size(x)..., size(y)...)))
end

"""
    table(fun::RankedDyad, x::AbstractArray, y::AbstractArray)

Table respects the left and right rank of the provided function and
computes `fun(x[i], y[j])` only on eachindex of the corresponding
frames of `x` and `y`.

# Example
Matrix product (note that the right argument is transposed)
```julia-repl
julia> table(ranked(1, (x, y) -> sum(x .* y), 1), reshape(1:6, 2, 3), reshape(1:12, 3, 4)')
2×4 Matrix{Int64}:
 22  49   76  103
 28  64  100  136
```
"""
function table(fd::RankedDyad, x::AbstractArray, y::AbstractArray)
    x = frame(x, max(ndims(x) - fd.leftrank, 0))
    y = frame(y, max(ndims(y) - fd.rightrank, 0))
    combine([fd.fun(x[i], y[j])
             for i in eachindex(x), j in eachindex(y)])
end

"""
    insert(fun, x)

Inserts `fun` between all items, i.e, sub-arrays sliced along the first dimension,
of `x`. As in J, associates `fun` to the right.

# Example
Same as `[1, 2, 3] + [4, 5, 6]`.
In contrast to `sum(+, [1 2 3; 4 5 6]; dims=1)` the
dimension of the resulting array is always reduced by one.
```julia-repl
julia> insert(+, [1 2 3; 4 5 6])
3-element Vector{Int64}:
 5
 7
 9
```

Same as `1 - (2 - 3)`
```julia-repl
julia> insert(-, [1, 2, 3])
2
```
"""
function insert(fun, x; init=Base._InitialValue())
    foldr(fun, frame(x, 1); init=init)
end
