
# Test of alternative implementation which ranks first dimensions
# Should be faster on column major arrays

using JuliennedArrays

function fastenclose end

fastenclose(x, rank::Val{M}) where {M}  = x

function fastenclose(data::AbstractArray{T,N}, rank::Val{M}) where {T,N,M}
    if M == 0
        data
    else
        Slices(data, ntuple(i -> if i > M False() else True() end, N)...)
    end
end

function fastcombine end

fastcombine(x) = x

function fastcombine(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    Align(parts, ntuple(i -> if i > I False() else True() end, I+O)...)
end

makeagree(x, y) = (x, y)

function _expand(x::AbstractArray{T,M}, n::Val{N}) where {T,M,N}
    sx = size(x)
    m = N - M
    reshape(x, ntuple(i -> if i > m sx[i-m] else 1 end, N))
end

function makeagree(left::AbstractArray{U,M}, right::AbstractArray{T,N}) where {U,M,T,N}
    NM = Val(max(N, M))
    _expand(left, NM), _expand(right, NM)
end

fbc2(fun, x, y) = broadcast(fun, makeagree(x, y)...)

struct FastRankedMonad{F,R}
    fun::F
end

function (fr::FastRankedMonad{F,R})(x) where {F,R}
    fastcombine(broadcast(fr.fun, fastenclose(x, Val(R))))
end

struct FastRankedDyad{F,L,R}
    fun::F
end

function (fd::FastRankedDyad{F,L,R})(x, y) where {F,L,R}
    left, right = makeagree(
        fastenclose(x, Val(L)),
        fastenclose(y, Val(R)))
    fastcombine(
        broadcast(
            fd.fun,
            left, right))
end

fastranked(fun, rank::Val{R}) where {R} = FastRankedMonad{typeof(fun),R}(fun)

fastranked(leftrank::Val{L}, fun, rightrank::Val{R}) where {L,R} = FastRankedDyad{typeof(fun),L,R}(fun)

macro rank_str(str)
    rx_dyad = r"^(\d+)\s+(.*)\s+(\d+)$"
    m_dyad = match(rx_dyad, str)
    if m_dyad == nothing
        # Try monad
        rx_monad = r"^(.*)\s+(\d+)$"
        m_monad = match(rx_monad, str)
        if m_monad == nothing
            error("Invalid format: must be <expr> <rank> or <leftrank> <expr> <rightrank>")
        else
            e, r = Meta.parse.(m_monad.captures)
            :(fastranked($(esc(e)), Val($r)))
        end
    else
        l, e, r = Meta.parse.(m_dyad.captures)
        :(fastranked(Val($l), $(esc(e)), Val($r)))
    end
end

fastiota(dims...) = reshape((1:prod(dims)) .- 1, dims)

fasttable(fun, x, y) = fun(x, y)

function _fasttable(fun, x::AbstractArray, y::AbstractArray)
    xl = reshape(x, (size(x)..., ntuple(i -> 1, ndims(y))...)...)
    yr = reshape(y, (ntuple(i -> 1, ndims(x))..., size(y)...)...)
    fastcombine(broadcast(fun, xl, yr))
end

function fasttable(fun, x::AbstractArray, y::AbstractArray)
    _fasttable(fun, x, y)
end

function fasttable(fd::FastRankedDyad{F,L,R}, x::AbstractArray, y::AbstractArray) where {F,L,R}
    _fasttable(fd.fun, fastenclose(x, Val(L)), fastenclose(y, Val(R)))
end

function fastinsert(fun, x::AbstractArray{T,N}; init=Base._InitialValue()) where {T,N}
    foldr(fun, fastenclose(x, Val(N-1)); init=init)
end

# Examples of this stuff

using Distributions
using Flux

reversedims(x::AbstractArray) = permutedims(x, ndims(x):-1:1)

fdist2(x, y) = sum((x .- y).^2)

fX = fastiota(4, 7)
fmu = fastiota(4, 3)
fd = fasttable(fastranked(Val(1), fdist2, Val(1)), fmu, fX)

fr = fbc2(==, fd, fastranked(x -> fastinsert(fastranked(Val(0), min, Val(0)), x), Val(1))(fd))

function fkmeans(fX, fmu)
    fd = fasttable(fastranked(Val(1), fdist2, Val(1)), fmu, fX)
    fr = fastranked(Val(0), ==, Val(0))(fd, fastranked(x -> fastinsert(fastranked(Val(0), min, Val(0)), x), Val(1))(fd))
    fastranked(Val(0), /, Val(0))(
        fastinsert(+, fastranked(Val(1), (x, y) -> fasttable(fastranked(Val(0), *, Val(0)), x, y), Val(1))(fX, fr)),
        fastinsert(+, fr))
end

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

struct FastAttentionHead{T}
    Wq::T
    Wk::T
    Wv::T
end

Flux.@functor FastAttentionHead

const EmbeddedTokens = AbstractMatrix

function (ah::FastAttentionHead)(y::EmbeddedTokens)
    q = ah.Wq * y
    k = ah.Wk * y
    v = ah.Wv * y
    # att = fastranked(softmax, Val(1))(fasttable(fastranked(Val(1), dot, Val(1)), k, q) ./ sqrt(size(q)[1]))
    att = rank"softmax 1"(fastranked(Val(2), (x, y) -> x' * y, Val(2))(k, q) ./ sqrt(size(q)[1]))
    v * att
end

struct FastMultiHead{U,T}
    Wproj::U
    heads::T
end

Flux.@functor FastMultiHead

function (mh::FastMultiHead)(y::EmbeddedTokens)
    res = rank"0 (h, x) -> h(x) 2"(mh.heads, y)  # apply all heads
    # Note: * at rank 2 acts as matrix multiplication
    fastinsert(rank"0 + 0", rank"2 * 2"(mh.Wproj, res))  # proj all res and sum
end

struct FastLayerNorm{U,T}
    shift::U
    scale::U
    eps::T
end

Flux.@functor FastLayerNorm
Flux.trainable(l::FastLayerNorm) = (l.shift, l.scale)

function (l::FastLayerNorm)(y::AbstractVector)
    # Again only defined on vector!
    l.shift .+ l.scale .* (y .- mean(y)) ./ (std(y; corrected=false) + l.eps)
end

struct MyFastTransformer{U,V,S,T}
    layernorm1::U
    multihead::V
    layernorm2::S
    mlp::T
end

Flux.@functor MyFastTransformer

function (trans::MyFastTransformer)(y::EmbeddedTokens)
    hidden = rank"trans.layernorm1 1"(y .+ trans.multihead(y))
    rank"trans.layernorm2 1"(hidden .+ rank"trans.mlp 1"(hidden))
end

function MyFastTransformer(t::Transformers.Transformer)
    # populate parameters from existing transformer layer
    laynorm1 = FastLayerNorm(t.mhn.diag.β, t.mhn.diag.α, t.mhn.ϵ)
    H = t.mh.head
    QH, D = size(t.mh.ikproj.weight)
    # Note: assuming bias of zero and identity σ
    Q = div(QH, H)
    mhs = FastAttentionHead.([reshape(t.mh.iqproj.weight, Q, H, D)[:, i, :] for i = 1:H],
                             [reshape(t.mh.ikproj.weight, Q, H, D)[:, i, :] for i = 1:H],
                             [reshape(t.mh.ivproj.weight, Q, H, D)[:, i, :] for i = 1:H])
    Wps = reshape(t.mh.oproj.weight, D, Q, H)
    laynorm2 = FastLayerNorm(t.pwn.diag.β, t.pwn.diag.α, t.pwn.ϵ)
    mlp = Chain(t.pw.din, t.pw.dout)
    MyFastTransformer(laynorm1, FastMultiHead(Wps, mhs), laynorm2, mlp)
end
    
myfasttrans = MyFastTransformer(trans)

@show size(fastranked(myfasttrans, Val(2))(batch))

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(fastcombine), parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    fastcombine(parts), Delta -> (NoTangent(), fastenclose(Delta, Val(I)))
end

function ChainRulesCore.rrule(::typeof(fastenclose), data::AbstractArray{T,N}, rank::Val{M}) where {T,N,M}
    fastenclose(data, rank), Delta -> (NoTangent(), fastcombine(Delta), ZeroTangent())
end
