
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
