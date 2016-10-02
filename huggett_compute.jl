#=
Program Name: huggett_compute.jl
=#

include("huggett.jl")
include("discretedp.jl")

test1 = Huggett()
testdp = DiscreteDP(test1.R,test1.Q,test1.beta)
#test3 = Huggett(q=0.9)
