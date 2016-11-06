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

function transition(steadystate_0,steadystate_N;N=30,epsilon=1e-2,
    epsilon_L=1e-2,max_iter=100,max_iter_L=100)

  # extract model results for beginning and ending steady states

  res_0 = steadystate_0[8]
  res_N = steadystate_N[8]

  # extract starting and ending K,L levels and prices r,w

  K_0 = steadystate_0[1]
  K_N = steadystate_N[1]

  L_0 = steadystate_0[2]
  L_N = steadystate_N[2]

  w_0 = steadystate_0[3]
  w_N = steadystate_N[3]

  r_0 = steadystate_0[4]
  r_N = steadystate_N[4]

  # initial guess for K sequence

  K_seq = linspace(K_0,K_N,N+1)

  # initialize sequences of L, w, r

  L_seq = zeros(Float64,N+1)
  L_seq[1] = L_0
  L_seq[N+1] = L_N

  r_seq = zeros(Float64,N)
  r_seq[1] = r_0
  r_seq[N+1] = r_N

  w_seq = zeros(Float64,N)
  w_seq[1] = w_0
  w_seq[N+1] = w_N

  for i in 1:N-1
    # define period
    t = N-i

    # initialize primitives for period t
    prim_t = Primitives(a_size=a_size)

    # take K_t from guessed sequence and guess L_t = L_t+1
    L_t = L_seq[t+1]

    # calculate prices given K_t, L_t (guess)
    prim_t.w = (1-prim_t.alpha)*K_seq[t]^(prim_t.alpha)*L_t^(-prim_t.alpha)
    prim_t.r = prim_t.alpha*K_seq[t]^(prim_t.alpha-1)*L_t^(1-prim_t.alpha) - prim_t.delta

    # solve program with initial parameters

    results_t = SolveProgram(prim_t)

    # iterate and update L_t until L_t is consistent with prices and K_t
    max_dist_L = 100.0

    for i in 1:max_iter_L

      # Print iteration, wage, rental rate
      println("Iter (L): ", i, " L: ", L_t, " Max Dist.: ", max_dist_L)

      # Calculate benefit
      mass_b = 0.00  # calculate mass receiving benefits
      for age in 1:prim_t.N-prim_t.JR+1
        mass_b += sum(results_t.ss_retired[:,age])
      end

      prim_t.b = (prim_t.theta*prim_t.w*L_t)/mass_b

      # Solve program given prices and benefit

      results_t = SolveProgram(prim_t)

      # Calculate new L_t

      L_t_new = 0.00
      for working_age in 1:prim_t.JR-1
        for asset in 1:prim_t.a_size
          L_t_new += results_t.ss_working_hi[asset,working_age]*
            results_t.labor_supply_hi[asset,working_age]*prim_t.ageeff[working_age]
          L_t_new += results_t.ss_working_lo[asset,working_age]*
            results_t.labor_supply_lo[asset,working_age]*prim_t.ageeff[working_age]
        end
      end

      # Adjust L_t if fails tolerance
      max_dist_L = max(abs(L_t-L_t_new))
      if max_dist_L < epsilon
          break
      else
        L_t_new = L_t*0.9 + L_t_new*0.1
      end

      # Calculate new prices
      w_new = (1-prim_t.alpha)*K_seq[t]^(prim_t.alpha)*L_t_new^(-prim_t.alpha)
      r_new = prim_t.alpha*K_seq[t]^(prim_t.alpha-1)*L_t_new^(1-prim_t.alpha) - prim_t.delta

      prim_t.w = w_new
      prim_t.r = r_new
      L_t = L_t_new

    end

    L_seq[t] = L_t

  end

end
