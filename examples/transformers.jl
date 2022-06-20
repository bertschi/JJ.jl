

# Transformer code ported from J

using Flux
using Distributions
using JJ

import Transformers

# create a small transformer layer and run it on an example

T = 5  # seq length, i.e., number of input tokens
D = 4  # embedding size
B = 8  # batch size
Q = 3  # attention head size
P = 6  # size of positiowise hidden layer

batch = rand(Normal(), D, T, B)

H = 2  # number of heads
trans = Transformers.Transformer(D, H, Q, P)

@show size(batch)
@show size(trans(batch))

# own implementation of transfomer layer

function softmax(y::AbstractVector)
    # Note: defined on vectors only!
    exp.(y .- logsumexp(y))
end

# dot product of vectors
dot(x::AbstractVector, y::AbstractVector) = sum(x .* y)

struct AttentionHead
    Wq
    Wk
    Wv
end

const EmbeddedTokens = AbstractMatrix

function (ah::AttentionHead)(y::EmbeddedTokens)
    q = y * ah.Wq
    k = y * ah.Wk
    v = y * ah.Wv
    att = ranked(softmax, 1)(table(ranked(1, dot, 1), q, k) ./ sqrt(size(q)[end]))
    # table(ranked(1, dot, 1), q, k) is same as q * k'
    att * v
end

struct MultiHead
    Wproj
    heads
end

function (mh::MultiHead)(y::EmbeddedTokens)
    res = ranked(0, (h, x) -> h(x), 2)(mh.heads, y)  # apply all heads
    # Note: * at rank 2 acts as matrix multiplication
    insert(.+, ranked(2, *, 2)(res, mh.Wproj))  # proj all res and sum
end

struct LayerNorm
    shift
    scale
    eps
end

function (l::LayerNorm)(y::AbstractVector)
    # Again only defined on vector!
    l.shift .+ l.scale .* (y .- mean(y)) ./ (std(y; corrected=false) + l.eps)
end

struct MyTransformer
    layernorm1
    multihead
    layernorm2
    mlp
end

function (trans::MyTransformer)(y::EmbeddedTokens)
    hidden = ranked(trans.layernorm1, 1)(y .+ trans.multihead(y))
    ranked(trans.layernorm2, 1)(hidden .+ ranked(trans.mlp, 1)(hidden))
end

function MyTransformer(t::Transformers.Transformer)
    # populate parameters from existing transformer layer
    laynorm1 = LayerNorm(t.mhn.diag.β, t.mhn.diag.α, t.mhn.ϵ)
    H = t.mh.head
    QH, D = size(t.mh.ikproj.weight)
    # Note: assuming bias of zero and identity σ
    Q = div(QH, H)
    mhs = AttentionHead.([reshape(t.mh.iqproj.weight, Q, H, D)[:, i, :]' for i = 1:H],
                         [reshape(t.mh.ikproj.weight, Q, H, D)[:, i, :]' for i = 1:H],
                         [reshape(t.mh.ivproj.weight, Q, H, D)[:, i, :]' for i = 1:H])
    Wps = permutedims(reshape(t.mh.oproj.weight, D, Q, H), (3, 2, 1))
    laynorm2 = LayerNorm(t.pwn.diag.β, t.pwn.diag.α, t.pwn.ϵ)
    mlp = Chain(t.pw.din, t.pw.dout)
    MyTransformer(laynorm1, MultiHead(Wps, mhs), laynorm2, mlp)
end
    
mytrans = MyTransformer(trans)

@show size(ranked(mytrans, 2)(permutedims(batch, (3, 2, 1))))
