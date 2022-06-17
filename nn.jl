# Small neural network example using JJ

using Distributions
using Flux
using Zygote
using ForwardDiff
using ReverseDiff
using ProgressBars
using JuliennedArrays
using ChainRulesCore

include("JJ.jl")

function nn(x, params, act)
    JJ.ranked(act, 0)(params.W * x + params.b)
end

# adgrad(fun, args::AbstractVector) = ReverseDiff.gradient(fun, args)
# adgrad(fun, args::AbstractVector) = ForwardDiff.gradient(fun, args)
adgrad(fun, args::AbstractVector) = Zygote.gradient(fun, args)[1]

function train_demo(x, y, params)
    ps, re = Flux.destructure(params)
    function loss(ps)
        pred = JJ.ranked(xi -> nn(xi, re(ps), tanh), 1)(x)
        err = JJ.ranked(1, (xi, yi) -> sum((xi .- yi).^2), 1)(pred, y)
        JJ.insert(+, err)
    end
    opt = Flux.Optimise.ADAM(1e-1)
    iter = ProgressBar(1:100)
    for i in iter
        sleep(0.1)
        grad = adgrad(loss, ps)
        Flux.Optimise.update!(opt, ps, grad)
        set_description(iter, "Loss: $(loss(ps))")
    end
    re(ps)
end

# Quick demo
N = 100
D = 5
Q = 2
W_true = rand(Normal(), Q, D)
b_true = rand(Normal(), Q)

X = randn(N, D)
Y = JJ.ranked(xi -> nn(xi, (W=W_true, b=b_true), tanh), 1)(X)

theta = (W = rand(Normal(0, 0.1), Q, D),
         b = rand(Normal(0, 0.1), Q))
train_demo(X, Y, theta)
