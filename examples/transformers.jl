

# Transformer code ported from J

using Flux
using Transformers
using Distributions

using JJ

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

# own implementation of (basic) transfomer layer

function softmax(y::AbstractVector)
    # Note: defined on vectors only!
    exp.(y .- logsumexp(y))
end

# dot product of vectors
dot(x::AbstractVector, y::AbstractVector) = sum(x .* y)

struct AttentionHead{T}
    Wq::T
    Wk::T
    Wv::T
end

Flux.@functor AttentionHead

const EmbeddedTokens = AbstractMatrix

function (ah::AttentionHead)(y::EmbeddedTokens)
    q = ah.Wq * y
    k = ah.Wk * y
    v = ah.Wv * y
    att = rank"softmax 1"(ranked(Val(2), (x, y) -> x' * y, Val(2))(k, q) ./ sqrt(size(q)[1]))
    v * att
end

struct MultiHead{U,T}
    Wproj::U
    heads::T
end

Flux.@functor MultiHead

function (mh::MultiHead)(y::EmbeddedTokens)
    res = rank"0 (h, x) -> h(x) 2"(mh.heads, y)  # apply all heads
    # Note: * at rank 2 acts as matrix multiplication
    insert(.+, rank"2 * 2"(mh.Wproj, res))  # proj all res and sum
end

struct LayerNorm{U,T}
    shift::U
    scale::U
    eps::T
end

Flux.@functor LayerNorm
Flux.trainable(l::LayerNorm) = (l.shift, l.scale)

function (l::LayerNorm)(y::AbstractVector)
    # Again only defined on vector!
    l.shift .+ l.scale .* (y .- mean(y)) ./ (std(y; corrected=false) + l.eps)
end

struct MyTransformer{U,V,S,T}
    layernorm1::U
    multihead::V
    layernorm2::S
    mlp::T
end

Flux.@functor MyTransformer

function (trans::MyTransformer)(y::EmbeddedTokens)
    hidden = rank"trans.layernorm1 1"(y .+ trans.multihead(y))
    rank"trans.layernorm2 1"(hidden .+ rank"trans.mlp 1"(hidden))
end

function MyTransformer(t::Transformers.Transformer)
    # populate parameters from existing transformer layer
    laynorm1 = LayerNorm(t.mhn.diag.β, t.mhn.diag.α, t.mhn.ϵ)
    H = t.mh.head
    QH, D = size(t.mh.ikproj.weight)
    # Note: assuming bias of zero and identity σ
    Q = div(QH, H)
    mhs = AttentionHead.([reshape(t.mh.iqproj.weight, Q, H, D)[:, i, :] for i = 1:H],
                         [reshape(t.mh.ikproj.weight, Q, H, D)[:, i, :] for i = 1:H],
                         [reshape(t.mh.ivproj.weight, Q, H, D)[:, i, :] for i = 1:H])
    Wps = reshape(t.mh.oproj.weight, D, Q, H)
    laynorm2 = LayerNorm(t.pwn.diag.β, t.pwn.diag.α, t.pwn.ϵ)
    mlp = Chain(t.pw.din, t.pw.dout)
    MyTransformer(laynorm1, MultiHead(Wps, mhs), laynorm2, mlp)
end

mytrans = MyTransformer(trans)

@show size(rank"mytrans 2"(batch))

all(trans(batch) .≈ rank"mytrans 2"(batch))
