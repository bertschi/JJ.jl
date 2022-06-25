
using ChainRulesCore

"""
    makeagree(x, y)

Reshapes its array arguments into a compatible shape by filling
missing dimensions with singletons from the front. This is in contrast
to standard broadcasting which fills missing dimensions at the end,
but is consistent with how `enframe` and `combine` work.

Note: No check is made if the existing dimensions agree -- before or
after expanding. Non matching shapes will simply lead to errors when
broadcasting later.

# Examples
```julia
julia> map(size, makeagree([1, 2], [1 2 3; 4 5 6]))
((1, 2), (2, 3))
```
"""

makeagree(x, y) = (x, y)

function _expand(x::AbstractArray{T,M}, n::Val{N}) where {T,M,N}
    sx = size(x)
    m = N - M
    reshape(x, ntuple(i -> if i > m sx[i-m] else 1 end, N))
end

function makeagree(left::AbstractArray{U,M}, right::AbstractArray{T,N}) where {U,M,T,N}
    NM = Val(max(N, M))
    _expand(left, NM), _expand(right, NM)
end

"""
    RankedMonad{F,R}(fun)

Represents a function `fun` of one argument at the given rank `R`.
"""
struct RankedMonad{F,R}
    fun::F
end

function (fr::RankedMonad{F,R})(x) where {F,R}
    combine(broadcast(fr.fun, enframe(x, Val(R))))
end

"""
    RankedDyad{F,L,R}(fun)

Represents a function `fun` of two arguments at the given left `L` and
right `R` ranks.
"""
struct RankedDyad{F,L,R}
    fun::F
end

function (fd::RankedDyad{F,L,R})(x, y) where {F,L,R}
    left, right = makeagree(
        enframe(x, Val(L)),
        enframe(y, Val(R)))
    combine(
        broadcast(
            fd.fun,
            left, right))
end

"""
    ranked(fun, rank)

Constructs a function acting like `fun` at `rank`.  The `rank` denotes
the size of sub-arrays which are passed to `fun` when applying
`ranked(fun, rank)` to a single array argument.

In order to allow compile-time optimizations the rank is passed as a
Val{N} type.

# Examples
Sum each row
```julia-repl
julia> ranked(sum, Val(1))([1 2 3; 4 5 6])
3-element Vector{Int64}:
 5
 7
 9
```
"""
ranked(fun, rank::Val{R}) where {R} = RankedMonad{typeof(fun),R}(fun)

"""
    ranked(leftrank, fun, rightrank)

Constructs a function acting like `fun` at `leftrank` and `rightrank`.
The ranks denote the sizes of sub-arrays from the left and right
arguments which are passed to `fun` when applying `ranked(leftrank,
fun, rightrank)` to two array arguments.

The function is then broadcasted across the left and right argument in
order to be applied to all pairs of sub-arrays. Thus, the shapes
remaining after constructing the sub-arrays have to agree (see
`makeagree` for details).

# Examples
```julia-repl
julia> ranked(Val(1), /, Val(0))([1 2 3; 4 5 6], [1, 2, 3])
2×3 Align{Float64, 2} with eltype Float64:
 1.0  1.0  1.0
 4.0  2.5  2.0
```

Like broadcasting, but aligning dimensions from the end.
```julia-repl
julia> ranked(Val(0), +, Val(0))([1, 2], reshape(1:6, 3, 2))
3×2 Matrix{Int64}:
 2  6
 3  7
 4  8
```
"""
ranked(leftrank::Val{L}, fun, rightrank::Val{R}) where {L,R} = RankedDyad{typeof(fun),L,R}(fun)

"""
    rank_str(s)

Parses its argument into <expr>␣<rank> or <leftrank>␣<expr>␣<rightrank>.

This syntax is an alternative to the `ranked` function and reminds of
the J operator \", makes ranked functions stand out and enforces that
ranks are compile-time integers.

# Examples
```julia-repl
julia> rank"1 / 0"([1 2 3; 4 5 6], [1, 2, 3])
2×3 Align{Float64, 2} with eltype Float64:
 1.0  1.0  1.0
 4.0  2.5  2.0
```
"""
macro rank_str(str)
    rx_dyad = r"^(\d+)\s+(.*)\s+(\d+)$"
    m_dyad = match(rx_dyad, str)
    if m_dyad == nothing
        # Try monad
        rx_monad = r"^(.*)\s+(\d+)$"
        m_monad = match(rx_monad, str)
        if m_monad == nothing
            error("Invalid format: must be <expr>␣<rank> or <leftrank>␣<expr>␣<rightrank>")
        else
            e, r = Meta.parse.(m_monad.captures)
            :(ranked($(esc(e)), Val($r)))
        end
    else
        l, e, r = Meta.parse.(m_dyad.captures)
        :(ranked(Val($l), $(esc(e)), Val($r)))
    end
end

# Rules to support Zygote's autodiff

function ChainRulesCore.rrule(::typeof(combine), parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    combine(parts), Delta -> (NoTangent(), enframe(Delta, Val(I)))
end

function ChainRulesCore.rrule(::typeof(enframe), data::AbstractArray{T,N}, rank::Val{M}) where {T,N,M}
    enframe(data, rank), Delta -> (NoTangent(), combine(Delta), ZeroTangent())
end
