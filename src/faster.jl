
# Test of alternative implementation which ranks first dimensions
# Should be faster on column major arrays

using JuliennedArrays

function fastframe end

fastframe(x, framerank::Int) = x

function fastframe(data::AbstractArray{T,N}, frameindex::Int) where {T,N}
    if frameindex < N
        Slices(data, ntuple(i -> if i > N - frameindex False() else True() end, N)...)
    else
        data
    end
end

function fastcombine end

fastcombine(x) = x

function fastcombine(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    Align(parts, ntuple(i -> if i > I False() else True() end, I+O)...)
end

makeagree(x, y) = (x, y)

function makeagree(left::AbstractArray{U,M}, right::AbstractArray{T,N}) where {U,M,T,N}
    leftsize = (ntuple(i -> 1, max(0, N-M))..., size(left)...)
    rightsize = (ntuple(i -> 1, max(0, M-N))..., size(right)...)
    reshape(left, leftsize), reshape(right, rightsize)
end

fbc2(fun, x, y) = broadcast(fun, makeagree(x, y)...)

struct FastRankedMonad{F,N}
    fun::F
    rank::N
end

function (fr::FastRankedMonad)(x)
    fastcombine(broadcast(fr.fun, fastframe(x, max(ndims(x) - fr.rank, 0))))
end

struct FastRankedDyad{F,M,N}
    fun::F
    leftrank::M
    rightrank::N
end

function (fd::FastRankedDyad)(x, y)
    left, right = makeagree(
        fastframe(x, max(ndims(x) - fd.leftrank, 0)),
        fastframe(y, max(ndims(y) - fd.rightrank, 0)))
    fastcombine(
        broadcast(
            fd.fun,
            left, right))
end

fastranked(fun, rank::Integer) = FastRankedMonad(fun, rank)

fastranked(leftrank::Integer, fun, rightrank::Integer) = FastRankedDyad(fun, leftrank, rightrank)

fastiota(dims...) = reshape((1:prod(dims)) .- 1, dims)

fasttable(fun, x, y) = fun(x, y)

function fasttable(fun, x::AbstractArray, y::AbstractArray)
    xl = reshape(x, (size(x)..., ntuple(i -> 1, ndims(y))...))
    yr = reshape(y, (ntuple(i -> 1, ndims(x))..., size(y)...))
    fastcombine(broadcast(fun, xl, yr))
end

function fasttable(fd::FastRankedDyad, x::AbstractArray, y::AbstractArray)
    xf = fastframe(x, max(ndims(x) - fd.leftrank, 0))
    yf = fastframe(y, max(ndims(y) - fd.rightrank, 0))
    xl = reshape(xf, (size(xf)..., ntuple(i -> 1, ndims(yf))...))
    yr = reshape(yf, (ntuple(i -> 1, ndims(xf))..., size(yf)...))
    fastcombine(broadcast(fd.fun, xl, yr))    
end

function fastinsert(fun, x; init=Base._InitialValue())
    foldr(fun, fastframe(x, 1); init=init)
end

# Examples of this stuff

reversedims(x::AbstractArray) = permutedims(x, ndims(x):-1:1)

fdist2(x, y) = sum((x .- y).^2)

fX = fastiota(4, 7)
fmu = fastiota(4, 3)
fd = fasttable(fdist2, fastframe(fmu, 1), fastframe(fX, 1))

fr = fbc2(==, fd, fastranked(x -> fastinsert(fastranked(0, min, 0), x), 1)(fd))

function fkmeans(fX, fmu)
    fd = fasttable(fastranked(1, fdist2, 1), fmu, fX)
    fr = fastranked(0, ==, 0)(fd, fastranked(x -> fastinsert(fastranked(0, min, 0), x), 1)(fd))
    fastranked(0, /, 0)(
        fastinsert(+, fastranked(1, (x, y) -> fasttable(fastranked(0, *, 0), x, y), 1)(fX, fr)),
        fastinsert(+, fr))
end
