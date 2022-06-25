
using JuliennedArrays

"""
    enframe(x, rank)

Splits argument `x` into `rank` dimensional array of sub-arrays --
starting from the front.  If `rank` is zero, `x` is just returned
unchanged.

In order to allow compile-time optimizations the rank is passed as a
Val{N} type.
"""
function enframe end

enframe(x, rank::Val{M}) where {M}  = x

function enframe(data::AbstractArray{T,N}, rank::Val{M}) where {T,N,M}
    if M == 0
        data
    else
        Slices(data, ntuple(i -> if i > M False() else True() end, N)...)
    end
end

"""
    combine(x)

Combines array of arrays into a larger array -- with the sub-array
dimensions first. All sub-arrays must have the same size! If `x` does
not contain any sub-arrays it is just returned unchanged.

# Examples
```julia-repl
julia> combine([[1, 2, 3], [4, 5, 6]])
3×2 Align{Int64, 2} with eltype Int64:
 1  4
 2  5
 3  6
```
"""
function combine end

combine(x) = x

function combine(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    Align(parts, ntuple(i -> if i > I False() else True() end, I+O)...)
end
