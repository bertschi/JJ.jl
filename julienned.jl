
using JuliennedArrays

frame(x, framerank::Int) = x

function frame(data::AbstractArray{T,N}, frameindex::Int) where {T,N}
    if frameindex < N
        Slices(data, ntuple(i -> if i > frameindex True() else False() end, N)...)
    else
        data
    end
end

combine(x) = x

function combine(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    Align(parts, ntuple(i -> if i > O True() else False() end, I+O)...)
end
