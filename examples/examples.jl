
using JJ

# Test some of this ... maybe on K-means

dist2(x, y) = sum((x .- y).^2)

X = iota(7, 4)
mu = iota(3, 4)
d = table(dist2, frame(X, 1), frame(mu, 1))

r = d .== ranked(x -> insert(ranked(0, min, 0), x), 1)(d)

function kmeans(X, mu)
    d = table(ranked(1, dist2, 1), X, mu)
    r = ranked(0, ==, 0)(d, ranked(x -> insert(ranked(0, min, 0), x), 1)(d))
    ranked(0, /, 0)(
        insert(+, ranked(1, (x, y) -> table(ranked(0, *, 0), x, y), 1)(r, X)),
        insert(+, r))
    # (r' * X) ./ sum(r; dims=1)'
end
