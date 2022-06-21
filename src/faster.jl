
# Test of alternative implementation which ranks first dimensions
# Should be faster on column major arrays

using JuliennedArrays

function fastframe end

fastframe(x, framerank::Int) = x

function fastframe(data::AbstractArray{T,N}, frameindex::Int) where {T,N}
    if frameindex < N
        Slices(data, ntuple(i -> if i > N - frameindex False() else True() end, N)...)
    else
        data
    end
end

function fastcombine end

fastcombine(x) = x

function fastcombine(parts::AbstractArray{<:AbstractArray{T,I}, O}) where {T,I,O}
    Align(parts, ntuple(i -> if i > I False() else True() end, I+O)...)
end

makeagree(x, y) = (x, y)

function makeagree(left::AbstractArray{U,M}, right::AbstractArray{T,N}) where {U,M,T,N}
    leftsize = (ntuple(i -> 1, max(0, N-M))..., size(left)...)
    rightsize = (ntuple(i -> 1, max(0, M-N))..., size(right)...)
    reshape(left, leftsize), reshape(right, rightsize)
end

fbc2(fun, x, y) = broadcast(fun, makeagree(x, y)...)

struct FastRankedMonad{F,N}
    fun::F
    rank::N
end

function (fr::FastRankedMonad)(x)
    fastcombine(broadcast(fr.fun, fastframe(x, max(ndims(x) - fr.rank, 0))))
end

struct FastRankedDyad{F,M,N}
    fun::F
    leftrank::M
    rightrank::N
end

function (fd::FastRankedDyad)(x, y)
    left, right = makeagree(
        fastframe(x, max(ndims(x) - fd.leftrank, 0)),
        fastframe(y, max(ndims(y) - fd.rightrank, 0)))
    fastcombine(
        broadcast(
            fd.fun,
            left, right))
end

fastranked(fun, rank::Integer) = FastRankedMonad(fun, rank)

fastranked(leftrank::Integer, fun, rightrank::Integer) = FastRankedDyad(fun, leftrank, rightrank)

fastiota(dims...) = reshape((1:prod(dims)) .- 1, dims)

fasttable(fun, x, y) = fun(x, y)

function fasttable(fun, x::AbstractArray, y::AbstractArray)
    xl = reshape(x, (size(x)..., ntuple(i -> 1, ndims(y))...))
    yr = reshape(y, (ntuple(i -> 1, ndims(x))..., size(y)...))
    fastcombine(broadcast(fun, xl, yr))
end

function fasttable(fd::FastRankedDyad, x::AbstractArray, y::AbstractArray)
    xf = fastframe(x, max(ndims(x) - fd.leftrank, 0))
    yf = fastframe(y, max(ndims(y) - fd.rightrank, 0))
    xl = reshape(xf, (size(xf)..., ntuple(i -> 1, ndims(yf))...))
    yr = reshape(yf, (ntuple(i -> 1, ndims(xf))..., size(yf)...))
    fastcombine(broadcast(fd.fun, xl, yr))    
end

function fastinsert(fun, x; init=Base._InitialValue())
    foldr(fun, fastframe(x, 1); init=init)
end

# Examples of this stuff

reversedims(x::AbstractArray) = permutedims(x, ndims(x):-1:1)

fdist2(x, y) = sum((x .- y).^2)

fX = fastiota(4, 7)
fmu = fastiota(4, 3)
fd = fasttable(fdist2, fastframe(fmu, 1), fastframe(fX, 1))

fr = fbc2(==, fd, fastranked(x -> fastinsert(fastranked(0, min, 0), x), 1)(fd))

function fkmeans(fX, fmu)
    fd = fasttable(fastranked(1, fdist2, 1), fmu, fX)
    fr = fastranked(0, ==, 0)(fd, fastranked(x -> fastinsert(fastranked(0, min, 0), x), 1)(fd))
    fastranked(0, /, 0)(
        fastinsert(+, fastranked(1, (x, y) -> fasttable(fastranked(0, *, 0), x, y), 1)(fX, fr)),
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

struct FastAttentionHead
    Wq
    Wk
    Wv
end

Flux.@functor FastAttentionHead

const EmbeddedTokens = AbstractMatrix

function (ah::FastAttentionHead)(y::EmbeddedTokens)
    q = ah.Wq * y
    k = ah.Wk * y
    v = ah.Wv * y
    att = fastranked(softmax, 1)(fasttable(fastranked(1, dot, 1), k, q) ./ sqrt(size(q)[end]))
    v * att
end

struct FastMultiHead
    Wproj
    heads
end

Flux.@functor FastMultiHead

function (mh::FastMultiHead)(y::EmbeddedTokens)
    res = fastranked(0, (h, x) -> h(x), 2)(mh.heads, y)  # apply all heads
    # Note: * at rank 2 acts as matrix multiplication
    fastinsert(fastranked(0, +, 0), fastranked(2, *, 2)(mh.Wproj, res))  # proj all res and sum
end

struct FastLayerNorm
    shift
    scale
    eps
end

Flux.@functor FastLayerNorm
Flux.trainable(l::FastLayerNorm) = (l.shift, l.scale)

function (l::FastLayerNorm)(y::AbstractVector)
    # Again only defined on vector!
    l.shift .+ l.scale .* (y .- mean(y)) ./ (std(y; corrected=false) + l.eps)
end

struct MyFastTransformer
    layernorm1
    multihead
    layernorm2
    mlp
end

Flux.@functor MyFastTransformer

function (trans::MyFastTransformer)(y::EmbeddedTokens)
    hidden = fastranked(trans.layernorm1, 1)(y .+ trans.multihead(y))
    fastranked(trans.layernorm2, 1)(hidden .+ fastranked(trans.mlp, 1)(hidden))
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

# @show size(ranked(mytrans, 2)(permutedims(batch, (3, 2, 1))))
