
using JuliennedArrays

"""
    frame(x, framerank)

Splits argument `x` into `framerank` dimensional array of sub-arrays.
If `framerank` is larger or equal to the dimension of `x` it is just
returned unchanged.
"""
function frame end

frame(x, framerank::Int) = x

function frame(data::AbstractArray{T,N}, frameindex::Int) where {T,N}
    if frameindex < N
        Slices(data, ntuple(i -> if i > frameindex True() else False() end, N)...)
    else
        data
    end
end

"""
    combine(x)

Combines array of arrays into a larger array. All sub-arrays must have
the same size! If `x` does not contain any sub-arrays it is just
returned unchanged.

# Examples
```julia-repl
julia> combine([[1, 2, 3], [4, 5, 6]])
2×3 Align{Int64, 2} with eltype Int64:
 1  2  3
 4  5  6
```
"""
function combine end

combine(x) = x

function combine(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    Align(parts, ntuple(i -> if i > O True() else False() end, I+O)...)
end

# define rules for AD

function ChainRulesCore.rrule(::Type{Slices{Item,Dimensions,Whole,Alongs}}, whole, alongs) where {Item,Dimensions,Whole,Alongs}
    Slices(whole, alongs...), Delta -> (NoTangent(), Align(Delta, alongs...), map(_ -> ZeroTangent(), alongs)...)
end

function ChainRulesCore.rrule(::Type{Align{Item,Dimensions,Sliced,Alongs}}, slices, alongs) where {Item,Dimensions,Sliced,Alongs}
    Align(slices, alongs...), Delta -> (NoTangent(), Slices(Delta, alongs...), map(_ -> ZeroTangent(), alongs)...)
end
