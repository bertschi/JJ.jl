
# Transformer code ported from J

using Flux
using Distributions

function logsoftmax(y::AbstractVector)
    y .- logsumexp(y)
end

dot(x::AbstractVector, y::AbstractVector) = sum(x .* y)

struct Head
    Wq
    Wk
    Wv
end

function (h::Head)(y)
    q = y * h.Wq
    k = y * h.Wk
    v = y * h.Wv
    att = ranked(logsoftmax, 1)(table(ranked(1, dot, 1), q, k) ./ sqrt(size(q)[end]))
    exp.(att) * v
end

struct MultiHead
    Wproj
    heads
end

function (m::MultiHead)(y)
    res = ranked(0, (h, x) -> h(x), 2)(m.heads, y)
    res = ranked(vec, 2)(permutedims(res, (2:length(size(res))..., 1)))
    res * m.Wproj
end

T = 5
D = 4
Q = 3
sentence = rand(Normal(), T, D)

function randhead(D, Q)
    Head(rand(Normal(), D, Q),
         rand(Normal(), D, Q),
         rand(Normal(), D, Q))
end

function randmultihead(H, D, Q)
    MultiHead(rand(Normal(), H*Q, D),
              [randhead(D, Q) for i in 1:H])
end

struct LayerNorm
    shift
    scale
end

function (l::LayerNorm)(y::AbstractVector)
    l.shift .+ l.scale .* (y .- mean(y)) ./ std(y)
end

struct MultiHeadAttention
    layernorm1
    multihead
    layernorm2
    mlp
end

function (mha::MultiHeadAttention)(y)
    hidden = ranked(mha.layernorm1, 1)(y .+ mha.multihead(y))
    ranked(mha.layernorm2, 1)(hidden .+ ranked(mha.mlp, 1)(hidden))
end

att = MultiHeadAttention(
    LayerNorm(0, 1),
    randmultihead(7, D, Q),
    LayerNorm(0, 1),
    Chain(Dense(D, 6, relu), Dense(6, D, relu))
)

batch = rand(Normal(), 16, T, D)

