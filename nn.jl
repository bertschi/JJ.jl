# Small neural network example using JJ

using Distributions
using Flux
using Zygote
using ForwardDiff
using ProgressBars
using JuliennedArrays

include("JJ.jl")

function nn(x, params, act)
    JJ.RankedMonad(act, 0)(params.W * x + params.b)
end

function train_demo(x, y, params)
    ps, re = Flux.destructure(params)
    function loss(ps)
        pred = JJ.RankedMonad(xi -> nn(xi, re(ps), tanh), 1)(x)
        err = JJ.RankedDyad((xi, yi) -> sum((xi .- yi).^2), 1, 1)(pred, y)
        JJ.insert(+, err)
    end
    opt = Flux.Optimise.ADAM(1e-1)
    iter = ProgressBar(1:100)
    for i in iter
        sleep(0.1)
        grad = ForwardDiff.gradient(loss, ps)
        Flux.Optimise.update!(opt, ps, grad)
        set_description(iter, "Loss: $(loss(ps))")
    end
    re(ps)
end

# TODO: Use ChainRules instead of Zygote adjoint

Zygote.@adjoint Slices(whole, alongs) =
    Slices(whole, alongs), Delta -> (Align(Delta, alongs), map(_ -> nothing, alongs)...)

Zygote.@adjoint Align(slices, alongs) =
    Align(slices, alongs), Delta -> (Slices(Delta, alongs), map(_ -> nothing, alongs)...)

Zygote.refresh()

# Zygote.@adjoint JJ.Framed(data, framerank) =
#     JJ.Framed(data, framerank), Delta -> (JJ.combine(Delta), nothing)

# framedim(::JJ.Combined{T,M,N,A}) where {T,M,N,A} = M
# framedim(x) = ndims(x)

# Zygote.@adjoint JJ.Combined(stuff) =
#     JJ.combine(stuff), Delta -> (JJ.frame(Delta, framedim(Delta)), )

# Zygote.refresh()

# Quick demo
N = 100
D = 5
Q = 2
W_true = rand(Normal(), Q, D)
b_true = rand(Normal(), Q)

X = randn(N, D)
Y = JJ.RankedMonad(xi -> nn(xi, (W=W_true, b=b_true), tanh), 1)(X)

train_demo(X,
           Y,
           (W = rand(Normal(0, 0.1), Q, D),
            b = rand(Normal(0, 0.1), Q)))

