module JJ

using Zygote
using ChainRulesCore

# include("framecombine.jl")
include("julienned.jl")
include("ranked.jl")
include("examples.jl")

Zygote.refresh()  # ensure that chain rules are visible

end # module
