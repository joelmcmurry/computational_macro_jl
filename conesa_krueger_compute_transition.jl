#=
Program Name: conesa_krueger_compute.jl
Runs Conesa-Krueger model and performs policy experiments
=#

using PyPlot
using LatexPrint

include("conesa_krueger_model.jl")

# Initialize primitives

prim = Primitives(a_size=300,a_max=30.0)

# Calculate relative cohort sizes
mu = ones(Float64,prim.N)
for j in 1:prim.N-1
  mu[j+1]=mu[j]/(1+prim.n)
end
# normalize so relative sizes sum to 1
mu = mu/(sum(mu))

# Solve model with social security

with_ss = compute_GE(a_size=prim.a_size,K0=1.99,L0=0.32)

# Without social security

without_ss = compute_GE(a_size=prim.a_size,theta=0.00,K0=2.5,L0=0.34)

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
  K_seq = linspace(K_0,K_T,T)
  L_seq = linspace(L_0,L_T,T)

  # initialize sequences of w, r
  r_seq = zeros(Float64,T)
  r_seq[1] = r_0
  r_seq[T] = r_T

  w_seq = zeros(Float64,T)
  w_seq[1] = w_0
  w_seq[T] = w_T

  # sequence of taxes
  theta_seq = zeros(Float64,T)
  theta_seq[1] = steadystate_0[9].theta
  theta_seq[T] = steadystate_T[9].theta

  max_dist = 100.0

  # store policy functions and labor supply each period
  policy_working_hi_seq = Array{Array{Int64}}(T)
  policy_working_lo_seq = Array{Array{Int64}}(T)
  policy_retired_seq = Array{Array{Int64}}(T)

  labor_supply_hi_seq = Array{Array{Float64}}(T)
  labor_supply_lo_seq = Array{Array{Float64}}(T)

  # initialize last policies
  policy_working_hi_seq[T] = res_T.policy_working_hi
  policy_working_lo_seq[T] = res_T.policy_working_lo
  policy_retired_seq[T] = res_T.policy_retired

  labor_supply_hi_seq[T] = res_T.labor_supply_hi
  labor_supply_lo_seq[T] = res_T.labor_supply_lo

  # initialize distributions over assets
  dist_working_hi_seq = Array{Array{Float64}}(T)
  dist_working_lo_seq = Array{Array{Float64}}(T)
  dist_retired_seq = Array{Array{Float64}}(T)

  for j in 1:max_iter

    # Print iteration and max distance between sequences
    println("Iter: ", j, " Max Dist.: ", max_dist)

    res_next = res_T
    for i in 1:T-1
      # define period
      t = T-i

      # initialize primitives for period t
      prim_t = Primitives(a_size=prim.a_size,theta=theta_seq[t])

      # calculate prices given K_t, L_t (guess)
      prim_t.w = (1-prim_t.alpha)*K_seq[t]^(prim_t.alpha)*L_seq[t]^(-prim_t.alpha)
      prim_t.r = prim_t.alpha*K_seq[t]^(prim_t.alpha-1)*L_seq[t]^(1-prim_t.alpha) - prim_t.delta

      # calculate benefit
      mass_b = 0.00  # calculate mass receiving benefits
      for age in 1:prim_t.N-prim_t.JR+1
        mass_b += sum(mu[age])
      end

      prim_t.b = (prim_t.theta*prim_t.w*L_seq[t])/mass_b

      # solve program with initial parameters
      results_t = SolveProgram_trans(prim_t,res_next)
      policy_retired_seq[t] = results_t.policy_retired
      policy_working_hi_seq[t] = results_t.policy_working_hi
      policy_working_lo_seq[t] = results_t.policy_working_lo

      labor_supply_hi_seq[t] = results_t.labor_supply_hi
      labor_supply_lo_seq[t] = results_t.labor_supply_lo

      res_next = results_t
    end

    # initialize distribution path with original steady state
    dist_working_hi_seq[1] = res_0.ss_working_hi
    dist_working_lo_seq[1] = res_0.ss_working_lo
    dist_retired_seq[1] = res_0.ss_retired
    for t in 2:T
      dist_working_hi_seq[t] = zeros(Float64,prim.a_size,prim.JR-1)
      dist_working_lo_seq[t] = zeros(Float64,prim.a_size,prim.JR-1)
      dist_retired_seq[t] = zeros(Float64,prim.a_size,prim.N-prim.JR+1)
    end

    #= Use policy rules to generate distribution path. In each period,
    newborn agents have zero assets and are distributed according to
    ergodic distribution. (Primitives and relative masses of each cohort
    are taken from first period of transition, but this is arbitrary
    as these objects are constant across all periods) =#
    tic()
    for t in 2:T
      dist_working_hi_seq[t][1,1] = mu[1]*prim.z_ergodic[1]
      dist_working_lo_seq[t][1,1] = mu[1]*prim.z_ergodic[2]
      for age in 2:prim.N
        for asset in 1:prim.a_size
          for choice_index in 1:prim.a_size
            if age < prim.JR # before retirement
              if policy_working_hi_seq[t-1][asset,age-1] == choice_index
                dist_working_hi_seq[t][choice_index,age] +=
                  (mu[age]/mu[age-1])*
                  dist_working_hi_seq[t-1][asset,age-1]*
                  prim.z_markov[1,1]
                dist_working_lo_seq[t][choice_index,age] +=
                  (mu[age]/mu[age-1])*
                  dist_working_hi_seq[t-1][asset,age-1]*
                  prim.z_markov[1,2]
              end
              if policy_working_lo_seq[t-1][asset,age-1] == choice_index
                dist_working_hi_seq[t][choice_index,age] +=
                  (mu[age]/mu[age-1])*
                  dist_working_lo_seq[t-1][asset,age-1]*
                  prim.z_markov[2,1]
                dist_working_lo_seq[t][choice_index,age] +=
                  (mu[age]/mu[age-1])*
                  dist_working_lo_seq[t-1][asset,age-1]*
                  prim.z_markov[2,2]
              end
            elseif age == prim.JR # at retirement
              if policy_working_hi_seq[t-1][asset,age-1] == choice_index
                dist_retired_seq[t][choice_index,1] +=
                  (mu[age]/mu[age-1])*dist_working_hi_seq[t-1][asset,age-1]
              end
              if policy_working_lo_seq[t-1][asset,age-1] == choice_index
                dist_retired_seq[t][choice_index,1] +=
                  (mu[age]/mu[age-1])*dist_working_lo_seq[t-1][asset,age-1]
              end
            else # retirement
              if policy_retired_seq[t-1][asset,age-prim.JR] == choice_index
                dist_retired_seq[t][choice_index,age-prim.JR+1] +=
                  (mu[age]/mu[age-1])*dist_retired_seq[t-1][asset,age-prim.JR]
              end
            end
          end
        end
      end
    end
    toc()

    # tic()
    # for t in 2:T
    #   dist_working_hi_seq[t][1,1] = mu[1]*prim.z_ergodic[1]
    #   dist_working_lo_seq[t][1,1] = mu[1]*prim.z_ergodic[2]
    #   for working_age in 2:prim.JR-1
    #     for asset in 1:prim.a_size
    #       for choice_index in 1:prim.a_size
    #         if policy_working_hi_seq[t-1][asset,working_age-1] == choice_index
    #           dist_working_hi_seq[t][choice_index,working_age] +=
    #             (mu[working_age]/mu[working_age-1])*
    #             dist_working_hi_seq[t-1][asset,working_age-1]*
    #             prim.z_markov[1,1]
    #           dist_working_lo_seq[t][choice_index,working_age] +=
    #             (mu[working_age]/mu[working_age-1])*
    #             dist_working_hi_seq[t-1][asset,working_age-1]*
    #             prim.z_markov[1,2]
    #         end
    #         if policy_working_lo_seq[t-1][asset,working_age-1] == choice_index
    #           dist_working_hi_seq[t][choice_index,working_age] +=
    #             (mu[working_age]/mu[working_age-1])*
    #             dist_working_lo_seq[t-1][asset,working_age-1]*
    #             prim.z_markov[2,1]
    #           dist_working_lo_seq[t][choice_index,working_age] +=
    #             (mu[working_age]/mu[working_age-1])*
    #             dist_working_lo_seq[t-1][asset,working_age-1]*
    #             prim.z_markov[2,2]
    #         end
    #       end
    #     end
    #   end
    #   for retired_age in 1:prim.N-prim.JR+1
    #     for asset in 1:prim.a_size
    #       for choice_index in 1:prim.a_size
    #         if retired_age == 1
    #           if policy_working_hi_seq[t-1][asset,prim.JR-1] == choice_index
    #             dist_retired_seq[t][choice_index,1] +=
    #               (mu[prim.JR]/mu[prim.JR-1])*dist_working_hi_seq[t-1][asset,prim.JR-1]
    #           end
    #           if policy_working_lo_seq[t-1][asset,prim.JR-1] == choice_index
    #             dist_retired_seq[t][choice_index,1] +=
    #               (mu[prim.JR]/mu[prim.JR-1])*dist_working_lo_seq[t-1][asset,prim.JR-1]
    #           end
    #         else
    #           if policy_retired_seq[t-1][asset,retired_age-1] == choice_index
    #               dist_retired_seq[t][choice_index,retired_age] +=
    #                 (mu[retired_age+prim.JR-1]/mu[retired_age+prim.JR-2])*dist_retired_seq[t-1][asset,retired_age-1]
    #           end
    #         end
    #       end
    #     end
    #   end
    # end
    # toc()

    # Calculate sequence of aggregate K and L

    K_seq_new = zeros(Float64,T)
    L_seq_new = zeros(Float64,T)
    for t in 1:T

      K_seq_new[t] = 0.00
      for working_age in 1:prim.JR-1
        for asset in 1:prim.a_size
          K_seq_new[t] += dist_working_hi_seq[t][asset,working_age]*prim.a_vals[asset]
          K_seq_new[t] += dist_working_lo_seq[t][asset,working_age]*prim.a_vals[asset]
        end
      end
      for retired_age in 1:prim.N-prim.JR+1
        for asset in 1:prim.a_size
          K_seq_new[t] += dist_retired_seq[t][asset,retired_age]*prim.a_vals[asset]
        end
      end

      L_seq_new[t] = 0.00
      for working_age in 1:prim.JR-1
        for asset in 1:prim.a_size
          L_seq_new[t] += dist_working_hi_seq[t][asset,working_age]*
            labor_supply_hi_seq[t][asset,working_age]*prim.ageeff[working_age]
          L_seq_new[t] += dist_working_lo_seq[t][asset,working_age]*
            labor_supply_lo_seq[t][asset,working_age]*prim.ageeff[working_age]
        end
      end
    end

    max_dist = max(maximum(abs(K_seq-K_seq_new)),maximum(abs(L_seq-L_seq_new)))

    if max_dist < epsilon
      break
    else
      K_seq = K_seq*0.9 + K_seq_new*0.1
      L_seq = L_seq*0.9 + L_seq_new*0.1
    end

  end

  # Recalculate price sequences
  for t in 1:T
    w_seq[t] = (1-prim.alpha)*K_seq[t]^(prim.alpha)*L_seq[t]^(-prim.alpha)
    r_seq[t] = prim.alpha*K_seq[t]^(prim.alpha-1)*L_seq[t]^(1-prim.alpha) - prim.delta
  end

  K_seq, L_seq, r_seq, w_seq, K_0, L_0, w_0, r_0, T

end

tic()
K_seq, L_seq, r_seq, w_seq, T = transition(with_ss,without_ss)
toc()

#= Graphs =#

Kfig = figure()
plot(linspace(0,T,T),K_seq,color="blue",linewidth=2.0)
ylabel("K")
legend(loc="lower right")
title("Capital Transition")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS6/Pictures/K_transition.pgf")

Lfig = figure()
plot(linspace(0,T,T),L_seq,color="red",linewidth=2.0)
ylabel("L")
legend(loc="lower right")
title("Labor Transition")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS6/Pictures/L_transition.pgf")

rfig = figure()
plot(linspace(0,T,T),r_seq,color="green",linewidth=2.0)
ylabel("r")
legend(loc="lower right")
title("Rental Rate Transition")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS6/Pictures/r_transition.pgf")

wfig = figure()
plot(linspace(0,T,T),w_seq,color="yellow",linewidth=2.0)
ylabel("w")
legend(loc="lower right")
title("Wage Transition")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS6/Pictures/r_transition.pgf")
