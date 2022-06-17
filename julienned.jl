
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
