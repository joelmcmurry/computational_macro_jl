#=
Program Name: conesa_krueger_compute.jl
Runs Conesa-Krueger model and performs policy experiments
=#

using PyPlot
using LatexPrint

include("conesa_krueger_model.jl")

# Define grid size and convergence iterations

a_size = 1000
max_iter = 100

# Solve model with social security

with_ss = compute_GE(a_size=a_size,max_iter=max_iter,K0=1.99,L0=0.32)

# Without social security

without_ss = compute_GE(a_size=a_size,max_iter=max_iter,theta=0.00,
  K0=2.5,L0=0.34)

#= Compute Transition Path =#

function transition(steadystate_0,steadystate_T;T=30,epsilon=1e-2,max_iter=100)

  # extract model results for beginning and ending steady states

  res_0 = steadystate_0[8]
  res_T = steadystate_T[8]

  # extract starting and ending K,L levels and prices r,w

  K_0 = steadystate_0[1]
  K_T = steadystate_T[1]

  L_0 = steadystate_0[2]
  L_T = steadystate_T[2]

  w_0 = steadystate_0[3]
  w_T = steadystate_T[3]

  r_0 = steadystate_0[4]
  r_T = steadystate_T[4]

  # initial guess for K,L sequences
  K_seq = linspace(K_0,K_T,T+1)
  L_seq = linspace(L_0,L_T,T+1)

  # initialize sequences of w, r
  r_seq = zeros(Float64,T+1)
  r_seq[1] = r_0
  r_seq[T+1] = r_T

  w_seq = zeros(Float64,T+1)
  w_seq[1] = w_0
  w_seq[T+1] = w_T

  for i in 1:T-1
    # define period
    t = T-i

    # initialize primitives for period t
    prim_t = Primitives(a_size=a_size)

    # calculate prices given K_t, L_t (guess)
    prim_t.w = (1-prim_t.alpha)*K_seq[t]^(prim_t.alpha)*L_seq[t]^(-prim_t.alpha)
    prim_t.r = prim_t.alpha*K_seq[t]^(prim_t.alpha-1)*L_seq[t]^(1-prim_t.alpha) - prim_t.delta

    # calculate benefit

      # find relative sizes of cohorts
      mu = ones(Float64,prim_t.N)
      for j in 1:prim_t.N-1
        mu[j+1]=mu[j]/(1+prim_t.n)
      end

      # Normalize so relative sizes sum to 1
      mu = mu/(sum(mu))
      mass_b = 0.00  # calculate mass receiving benefits
      for age in 1:prim_t.N-prim_t.JR+1
        mass_b += sum(mu[age])
      end

    prim_t.b = (prim_t.theta*prim_t.w*L_seq[t])/mass_b

    # solve program with initial parameters

    results_t = SolveProgram(prim_t,steadyflag="no")

  end

end
