module JJ

using JuliennedArrays

tally(x::Tuple) = length(x)
tally(x::AbstractArray) = size(x, 1)

shape(x::AbstractArray) = size(x)

rank(x::AbstractArray) = tally(shape(x))
rank(x) = 0

function ranked(fun, r::Integer, x::AbstractArray)
    n = rank(x)
    r = min(r, n)  # effective rank
    f = n - r  # frame rank
    if f == 0
        fun(x)
    else
        inner = f .+ (1:r)
        split = Slices(x, inner...)
        res = map(fun, split)
        # find new inner dimension
        ri = rank(res[eachindex(res)[1]])
        if ri > 0
            Align(res, (f .+ (1:ri))...)
        else
            res
        end
    end
end

# Try using own type for array slicing

struct Framed{T,M,N,A} <: AbstractArray{T,M}
    data::A
end

function Framed{T,M}(data::AbstractArray{T,N}) where {T,M,N}
    @assert M <= N
    Framed{T,M,N,typeof(data)}(data)
end

function Framed(data::AbstractArray, framerank::Int)
    Framed{eltype(data),framerank}(data)
end

Base.axes(A::Framed{T,M}) where {T,M} = axes(A.data)[1:M]

Base.size(A::Framed) = map(length, axes(A))

function drop(x::Tuple, n::Int)
    @assert n >= 0
    if n == 0
        x
    else
        drop(Base.tail(x), n-1)
    end
end

function Base.getindex(A::Framed{T,M}, I::Vararg{Int,M}) where {T,M}
    view(A.data, I..., drop(axes(A.data), M)...)
end

struct Combined{T,M,N,A} <: AbstractArray{T,N}
    parts::A
end

function check_align(partsizes)
    if !isempty(partsizes)
        firstsize = first(partsizes)
        @assert all(x == firstsize for x in partsizes)
    end
end

function Combined(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    check_align(map(size, parts))
    Combined{T,O,I+O,typeof(parts)}(parts)
end

Base.axes(x::Combined) = (axes(x.parts)..., axes(first(x.parts))...)

Base.size(x::Combined) = map(length, axes(x))

function Base.getindex(x::Combined{T,M,N,A}, I::Vararg{Int,N}) where {T,M,N,A}
    x.parts[I[1:M]...][I[(M+1):end]...]
end

function ranked(fun, r::Integer, x::AbstractArray)
    Combined(map(fun, Framed(x, max(ndims(x) - r, 0))))
end
    
end # module
