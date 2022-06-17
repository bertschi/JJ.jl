
# Test some of this ... maybe on K-means

dist2(x, y) = sum((x .- y).^2)

X = iota(4, 7)'
mu = iota(4, 3)'
d = table(dist2, frame(X, 1), frame(mu, 1))

bc(fun) = (args...) -> fun.(args...)

r = d .== ranked(x -> insert(bc(min), x), 1, d)

function kmeans(X, mu)
    d = table(dist2, frame(X, 1), frame(mu, 1))
    r = d .== ranked(x -> insert(bc(min), x), 1, d)
    insert(+, combine(broadcast((x, y) -> table(bc(*), x, y), frame(r, 1), frame(X, 1)))) ./ insert(+, r)
    # (r' * X) ./ sum(r; dims=1)'
end

# seems to work, but is not very nice (yet)

function kmeans2(X, mu)
    d = table(RankedDyad(dist2, 1, 1), X, mu)
    r = d .== RankedMonad(x -> insert(RankedDyad(min, 0, 0), x), 1)(d)
    RankedDyad(/, 0, 0)(
        insert(+, RankedDyad((x, y) -> table(RankedDyad(*, 0, 0), x, y), 1, 1)(r, X)),
        insert(+, r))
    # (r' * X) ./ sum(r; dims=1)'
end
