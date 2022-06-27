
using JJ

# Some examples using JJ

################################################################################
# Different ways to implement matrix multiplication

A = reshape(1:6, 2, 3)
B = reshape(1:12, 3, 4)

@show A * B  # standard matrix multiplication
# In J notation: A +/ . * B

@show rank"x -> insert(+, x) 1"(table(rank"1 rank\"0 * 0\" 1", A', B))
# In J notation: +/"1 A *"1/ |: B

# or slightly simpler using partial application and broadcasting instead of rank 0 function
partial(f, args...) = (moreargs...) -> f(args..., moreargs...)
@show rank"partial(insert, +) 1"(table(rank"1 .* 1", A', B))

@show insert(+, rank"1 partial(table, .*) 1"(A, B'))
# In J notation: +/ (|: A) */"1 B

################################################################################
# K-Means clustering

dist2(x, y) = sum((x .- y).^2)

function kmeans(X, mu)
    d = table(rank"1 dist2 1", mu, X)
    # Note: .op and rank"0 op 0" are not the same as arguments are broadcasted differently
    r = rank"0 == 0"(d, rank"partial(insert, min) 1"(d))
    rank"0 / 0"(
        insert(+, rank"1 partial(table, *) 1"(X, r)),
        insert(+, r))
end

X = reshape(1:28, 4, 7)
mu = reshape(1:12, 4, 3)

@show kmeans(X, mu)
