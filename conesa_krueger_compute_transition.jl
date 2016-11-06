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

function transition(steadystate_0,steadystate_N;N=30,epsilon=1e-2)

  # extract model results for beginning and ending steady states

  res_0 = steadystate_0[8]
  res_N = steadystate_N[8]

  # extract starting and ending K,L levels

  K_0 = steadystate_0[1]
  K_N = steadystate_N[1]

  L_0 = steadystate_0[2]
  L_N = steadystate_N[2]

  # initial guess for K sequence

  K_seq = linspace(K_0,K_N,N)

  # initialize sequence of L

  L_seq = zeros(Float64,N)
  L_seq[N] = L_N

  for i in 1:N-1
    # define period
    t = N-i

    # initialize primitives and results for period t
    prim = Primitives(a_size)
    res = Results(prim)

    

    #= guess L_t = L_t+1 and calculate prices and aggregate labor
    given K_t, L_t. Find L_t consistent with prices =#
    prim.w = (1-prim.alpha)*K_seq[t]^(prim.alpha)*L_seq[t+1]^(-prim.alpha)
    prim.r = prim.alpha*K_seq[t]^(prim.alpha-1)*L_seq[t+1]^(1-prim.alpha) - prim.delta


    L_new = 0.00
    for working_age in 1:prim.JR-1
      for asset in 1:prim.a_size
        L_new += results.ss_working_hi[asset,working_age]*
          results.labor_supply_hi[asset,working_age]*prim.ageeff[working_age]
        L_new += results.ss_working_lo[asset,working_age]*
          results.labor_supply_lo[asset,working_age]*prim.ageeff[working_age]
      end
    end

    # Adjust K, L if fails tolerance
    max_dist = max(abs(K-K_new),abs(L-L_new))
    if max_dist < epsilon
        break
    else
      L_new = L*0.9 + L_new*0.1
      K_new = K*0.9 + K_new*0.1
    end


  end

end
