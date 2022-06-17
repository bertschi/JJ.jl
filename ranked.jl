
function ranked(fun, r::Integer, x::AbstractArray)
    combine(broadcast(fun, frame(x, max(ndims(x) - r, 0))))
end

iota(dims...) = reshape((1:prod(dims)) .- 1, dims)  # Order not matching J

function table(fun, x, y)
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
    ranked(fr.fun, fr.rank, x)
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

function table(fd::RankedDyad, x, y)
    x = frame(x, max(ndims(x) - fd.leftrank, 0))
    y = frame(y, max(ndims(y) - fd.rightrank, 0))
    combine([fd.fun(x[i], y[j])
             for i in eachindex(x), j in eachindex(y)])
end
