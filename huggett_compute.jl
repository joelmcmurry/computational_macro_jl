#=
Program Name: huggett_compute.jl
=#

include("huggett.jl")
include("discretedp.jl")

huggett = Huggett()

huggettdp = DiscreteProgram(huggett.R,huggett.Q,huggett.beta)

huggettres = SolveProgram(huggettdp,max_iter=2000)
