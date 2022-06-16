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

function Combined(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    check_align(map(size, parts))
    Combined{T,O,I+O,typeof(parts)}(parts)
end

function Combined(parts::AbstractArray{T,N}) where {T,N}
    # Combined{T,N,N,typeof(parts)}(parts)
    parts
end

Combined(parts) = parts  # special case for scalars

Base.axes(x::Combined) = (axes(x.parts)..., axes(first(x.parts))...)

Base.size(x::Combined) = map(length, axes(x))

function Base.getindex(x::Combined{T,M,N,A}, I::Vararg{Int,N}) where {T,M,N,A}
    x.parts[I[1:M]...][I[(M+1):end]...]
end

function ranked(fun, r::Integer, x::AbstractArray)
    Combined(broadcast(fun, frame(x, max(ndims(x) - r, 0))))
end

# Test some of this ... maybe on K-means

dist2(x, y) = sum((x .- y).^2)

iota(dims...) = reshape((1:prod(dims)) .- 1, dims)  # Order not matching J

function table(fun, x, y)
    Combined([fun(x[i], y[j])
              for i in eachindex(x), j in eachindex(y)])
end

X = iota(4, 7)'
mu = iota(4, 3)'
d = table(dist2, frame(X, 1), frame(mu, 1))

function insert(fun, x)
    reduce(fun, frame(x, 1))
end

bc(fun) = (args...) -> fun.(args...)

r = d .== ranked(x -> insert(bc(min), x), 1, d)

function kmeans(X, mu)
    d = table(dist2, frame(X, 1), frame(mu, 1))
    r = d .== ranked(x -> insert(bc(min), x), 1, d)
    insert(+, Combined(broadcast((x, y) -> table(bc(*), x, y), frame(r, 1), frame(X, 1)))) ./ insert(+, r)
    # (r' * X) ./ sum(r; dims=1)'
end

# seems to work, but is not very nice (yet)

# Let's try with type for Ranked function

struct RankedMonad{F,N}
    fun::F
    rank::N
end

function (fr::RankedMonad)(x)
    ranked(fr.fun, fr.rank, x)
end

struct RankedDyad{F,M,N}
    fun::F
    leftrank::M
    rightrank::N
end

function (fd::RankedDyad)(x, y)
    Combined(
        broadcast(
            fd.fun,
            frame(x, max(ndims(x) - fd.leftrank, 0)),
            frame(y, max(ndims(y) - fd.rightrank, 0))))
end

function kmeans2(X, mu)
    d = table(dist2, frame(X, 1), frame(mu, 1))
    r = d .== RankedMonad(x -> insert(RankedDyad(min, 0, 0), x), 1)(d)
    RankedDyad(/, 0, 0)(
        insert(+, RankedDyad((x, y) -> table(RankedDyad(*, 0, 0), x, y), 1, 1)(r, X)),
        insert(+, r))
    # (r' * X) ./ sum(r; dims=1)'
end

end # module
