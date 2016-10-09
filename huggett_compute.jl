#=
Program Name: huggett_compute.jl
=#

include("huggett.jl")
include("discretedp.jl")

## Instantiate Huggett model

huggett = Huggett()

## Create Dynamic Program

huggettdp = DiscreteProgram(huggett.R,huggett.Q,huggett.beta)

## Solve Dynamic Program to yield value function and policy function

huggettres = SolveProgram(huggettdp,max_iter_vfi=2000)

## Policy function is in terms of indices. Extract asset holding rule

policy_vals = huggett.a_vals[huggettres.sigma]
