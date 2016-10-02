#=
Program Name: huggett_compute.jl
=#

using QuantEcon
include("huggett.jl")
include("discretedp.jl")

test1 = Huggett()
test2 = DiscreteDP(test1.R,test1.Q,test1.beta)
#test3 = Huggett(q=0.9)
