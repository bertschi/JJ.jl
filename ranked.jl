
function iota(dims...)
    permutedims(
        reshape((1:prod(dims)) .- 1, reverse(dims)),
        reverse(1:length(dims)))
end

table(fun, x, y) = fun(x, y)  # scalar default

function table(fun, x::AbstractArray, y::AbstractArray)
    combine([fun(x[i], y[j])
             for i in eachindex(x), j in eachindex(y)])
end

function insert(fun, x)
    reduce(fun, frame(x, 1))
end

# Let's try with type for Ranked function

struct RankedMonad{F,N}
    fun::F
    rank::N
end

function (fr::RankedMonad)(x)
    combine(broadcast(fr.fun, frame(x, max(ndims(x) - fr.rank, 0))))
end

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

ranked(fun, rank::Integer) = RankedMonad(fun, rank)
ranked(leftrank::Integer, fun, rightrank::Integer) = RankedDyad(fun, leftrank, rightrank)

function table(fd::RankedDyad, x::AbstractArray, y::AbstractArray)
    x = frame(x, max(ndims(x) - fd.leftrank, 0))
    y = frame(y, max(ndims(y) - fd.rightrank, 0))
    combine([fd.fun(x[i], y[j])
             for i in eachindex(x), j in eachindex(y)])
end
