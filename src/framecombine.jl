# Try using own type for array slicing

struct Framed{T,M,N,A} <: AbstractArray{T,M}
    data::A
end

function Framed{ET,M}(data::AbstractArray{T,N}) where {ET,M,N,T}
    @assert M < N
    Framed{ET,M,N,typeof(data)}(data)
end

function frameindex(ax, I)
    M = length(I)
    ntuple(i -> if i > M ax[i] else I[i] end, length(ax))
end

function Framed(data::AbstractArray, framerank::Int)
    ET = typeof(view(data, frameindex(axes(data), ntuple(i->1, framerank))...))
    Framed{ET,framerank}(data)
end

Base.axes(A::Framed{T,M}) where {T,M} = axes(A.data)[1:M]

Base.size(A::Framed) = map(length, axes(A))

function Base.getindex(A::Framed{T,M}, I::Vararg{Int,M}) where {T,M}
    view(A.data, frameindex(axes(A.data), I)...)
end

function frame(data::AbstractArray{T,N}, framerank::Int) where {T,N}
    if framerank >= N
        data
    else
        Framed(data, framerank)
    end
end

frame(data, framerank::Int) = data  # scalars stay themselves at any rank!

struct Combined{T,M,N,A} <: AbstractArray{T,N}
    parts::A
end

function check_align(partsizes)
    if !isempty(partsizes)
        firstsize = first(partsizes)
        @assert all(x == firstsize for x in partsizes)
    end
end

function combine(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    check_align(map(size, parts))
    Combined{T,O,I+O,typeof(parts)}(parts)
end

function combine(parts::AbstractArray{T,N}) where {T,N}
    # Combined{T,N,N,typeof(parts)}(parts)
    parts
end

combine(parts) = parts  # special case for scalars

Base.axes(x::Combined) = (axes(x.parts)..., axes(first(x.parts))...)

Base.size(x::Combined) = map(length, axes(x))

function Base.getindex(x::Combined{T,M,N,A}, I::Vararg{Int,N}) where {T,M,N,A}
    x.parts[I[1:M]...][I[(M+1):end]...]
end

# define rules for AD

function ChainRulesCore.rrule(::Type{Framed{T,M,N,A}}, data, framerank::Int) where {T,M,N,A}
    frame(data, framerank), Delta -> (NoTangent(), combine(Delta), ZeroTangent())
end

function ChainRulesCore.rrule(::Type{Combined{T,M,N,A}}, parts) where {T,M,N,A}
    combine(parts), Delta -> (NoTangent(), frame(Delta, M))
end
