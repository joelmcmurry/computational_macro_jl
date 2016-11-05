#=
Program Name: conesa_krueger_compute.jl
Runs Conesa-Krueger model and performs policy experiments
=#

using PyPlot
using LatexPrint

include("conesa_krueger_model.jl")

# Solve model with social security

with_ss = compute_GE(a_size=1000,max_iter=100,K0=1.99,L0=0.32)

# Without social security

without_ss = compute_GE(a_size=1000,max_iter=100,theta=0.00,
  K0=2.5,L0=0.34)
