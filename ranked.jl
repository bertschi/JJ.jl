
"""
    RankedMonad(fun, rank)

Represents a function `fun` of one argument at the given `rank`.
"""
struct RankedMonad{F,N}
    fun::F
    rank::N
end

function (fr::RankedMonad)(x)
    combine(broadcast(fr.fun, frame(x, max(ndims(x) - fr.rank, 0))))
end

"""
    RankedDyad(fun, leftrank, rightrank)

Represents a function `fun` of two arguments at the given `leftrank` and `rightrank`.
"""
struct RankedDyad{F,M,N}
    fun::F
    leftrank::M
    rightrank::N
end

function (fd::RankedDyad)(x, y)
    combine(
        broadcast(
            fd.fun,
            frame(x, max(ndims(x) - fd.leftrank, 0)),
            frame(y, max(ndims(y) - fd.rightrank, 0))))
end

"""
    ranked(fun, rank)

Constructs a function acting like `fun` at `rank`.  The `rank` denotes
the size of sub-arrays which are passed to `fun` when applying
`ranked(fun, rank)` to a single array argument.

# Examples
Sum each row
```julia-repl
julia> ranked(sum, 1)([1 2 3; 4 5 6])
2-element Vector{Int64}:
  6
 15
```
"""
ranked(fun, rank::Integer) = RankedMonad(fun, rank)

"""
    ranked(fun, rank)

Constructs a function acting like `fun` at `leftrank` and `rightrank`.
The ranks denote the sizes of sub-arrays from the left and right
arguments which are passed to `fun` when applying `ranked(leftrank,
fun, rightrank)` to two array arguments.

# Examples
```julia-repl
julia> ranked(1, /, 0)([1 2 3; 4 5 6], [1, 2])
2×3 Align{Float64, 2} with eltype Float64:
 1.0  2.0  3.0
 2.0  2.5  3.0
```
"""
ranked(leftrank::Integer, fun, rightrank::Integer) = RankedDyad(fun, leftrank, rightrank)
