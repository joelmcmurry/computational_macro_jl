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

#= General Equilibrium =#

function compute_GE(;a_size=100,theta=0.11,z_vals=[3.0, 0.5],gamma=0.42,
    epsilon=1e-2,max_iter=100,K0::Float64=2.0,L0::Float64=0.3)

  # Initialize primitives
  prim = Primitives(a_size=a_size,theta=theta,gamma=gamma,z_vals=z_vals)

  # Solve problem with default values
  results = SolveProgram(prim)

  # Initialize aggregate capital and labor with Initial guess of capital and labor

  K = K0
  L = L0

  # Calculate initial wages and rental rate
  prim.w = (1-prim.alpha)*K^(prim.alpha)*L^(-prim.alpha)
  prim.r = prim.alpha*K^(prim.alpha-1)*L^(1-prim.alpha) - prim.delta

  max_dist = 100.00

  for i in 1:max_iter

    # Print iteration, wage, rental rate
    println("Iter: ", i, " L: ", L," K: ", K, " Max Dist.: ", max_dist)

    # Calculate benefit
    mass_b = 0.00  # calculate mass receiving benefits
    for age in 1:prim.N-prim.JR+1
      mass_b += sum(results.ss_retired[:,age])
    end

    prim.b = (prim.theta*prim.w*L)/mass_b

    # Solve program given prices and benefit

    results = SolveProgram(prim)

    # Calculate new aggregate capital and labor

    K_new = 0.00
    for working_age in 1:prim.JR-1
      for asset in 1:prim.a_size
        K_new += results.ss_working_hi[asset,working_age]*
          prim.a_vals[results.policy_working_hi[asset,working_age]]
        K_new += results.ss_working_lo[asset,working_age]*
          prim.a_vals[results.policy_working_lo[asset,working_age]]
      end
    end
    for retired_age in 1:prim.N-prim.JR+1
      for asset in 1:prim.a_size
        if results.policy_retired[asset,retired_age] != 0
          K_new += results.ss_retired[asset,retired_age]*
            prim.a_vals[results.policy_retired[asset,retired_age]]
        end
      end
    end

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

    # Calculate new prices
    w_new = (1-prim.alpha)*K_new^(prim.alpha)*L_new^(-prim.alpha)
    r_new = prim.alpha*K_new^(prim.alpha-1)*L_new^(1-prim.alpha) - prim.delta

    prim.w = w_new
    prim.r = r_new
    L = L_new
    K = K_new

  end

  K, L, prim.w, prim.r, prim.b, results.W, results.cv, results, prim

end

#= Policy Experiments =#

## Baseline: idiosynractic risk and endogenous labor

function baseline_calc(;a_size=1000,max_iter=100)
  # with social security
  with_ss = compute_GE(a_size=a_size,max_iter=max_iter,K0=1.99,L0=0.32)
  # without social security
  without_ss = compute_GE(a_size=a_size,max_iter=max_iter,theta=0.00,
    K0=2.5,L0=0.34)

    prim = with_ss[9]
    res = with_ss[8]
    res_no_ss = without_ss[8]

    vote_yes = 0.0
    for working_age in 1:prim.JR-1
      for asset in 1:prim.a_size
        if res.v_working_hi[asset,working_age] < res_no_ss.v_working_hi[asset,working_age]
          vote_yes += res.ss_working_hi[asset,working_age]
        end
        if res.v_working_lo[asset,working_age] < res_no_ss.v_working_lo[asset,working_age]
          vote_yes += res.ss_working_lo[asset,working_age]
        end
      end
    end
    for retired_age in 1:prim.N-prim.JR+1
      for asset in 1:prim.a_size
        if res.v_retired[asset,retired_age] < res_no_ss.v_retired[asset,retired_age]
          vote_yes += res.ss_retired[asset,retired_age]
        end
      end
    end

  return with_ss, without_ss, vote_yes
end
tic()
baseline, baseline_no_ss, baseline_vote_yes = baseline_calc()
toc()

## No idiosyncratic risk

function no_idio_risk_calc(;a_size=1000,max_iter=100)
  # with social security
  with_ss = compute_GE(a_size=a_size,max_iter=max_iter,z_vals=[0.5,0.5],
    K0=1.99,L0=0.32)
  # without social security
  without_ss = compute_GE(a_size=a_size,max_iter=max_iter,z_vals=[0.5,0.5],
    theta=0.00,K0=2.5,L0=0.34)

    prim = with_ss[9]
    res = with_ss[8]
    res_no_ss = without_ss[8]

    vote_yes = 0.0
    for working_age in 1:prim.JR-1
      for asset in 1:prim.a_size
        if res.v_working_hi[asset,working_age] < res_no_ss.v_working_hi[asset,working_age]
          vote_yes += res.ss_working_hi[asset,working_age]
        end
        if res.v_working_lo[asset,working_age] < res_no_ss.v_working_lo[asset,working_age]
          vote_yes += res.ss_working_lo[asset,working_age]
        end
      end
    end
    for retired_age in 1:prim.N-prim.JR+1
      for asset in 1:prim.a_size
        if res.v_retired[asset,retired_age] < res_no_ss.v_retired[asset,retired_age]
          vote_yes += res.ss_retired[asset,retired_age]
        end
      end
    end

  return with_ss, without_ss, vote_yes
end
tic()
no_idio_risk, no_idio_risk_no_ss, no_idio_risk_vote_yes = no_idio_risk_calc()
toc()

## Exogenous Labor

function exo_labor_calc(;a_size=1000,max_iter=100)
  # with social security
  with_ss = compute_GE(a_size=1000,max_iter=100,gamma=1.00,K0=5.0,L0=0.75)
  # without social security
  without_ss = compute_GE(a_size=1000,max_iter=100,gamma=1.00,
    theta=0.00,K0=6.3,L0=0.75)

    prim = with_ss[9]
    res = with_ss[8]
    res_no_ss = without_ss[8]

    vote_yes = 0.0
    for working_age in 1:prim.JR-1
      for asset in 1:prim.a_size
        if res.v_working_hi[asset,working_age] < res_no_ss.v_working_hi[asset,working_age]
          vote_yes += res.ss_working_hi[asset,working_age]
        end
        if res.v_working_lo[asset,working_age] < res_no_ss.v_working_lo[asset,working_age]
          vote_yes += res.ss_working_lo[asset,working_age]
        end
      end
    end
    for retired_age in 1:prim.N-prim.JR+1
      for asset in 1:prim.a_size
        if res.v_retired[asset,retired_age] < res_no_ss.v_retired[asset,retired_age]
          vote_yes += res.ss_retired[asset,retired_age]
        end
      end
    end

  return with_ss, without_ss, vote_yes
end
tic()
exo_labor, exo_labor_no_ss, exo_labor_vote_yes = exo_labor_calc()
toc()

#= Tables for Output to LaTeX =#

output = Array(Any,(9,7))
titles_vert = ["K","L","w","r","b","W","cv","vote"]
titles_horz = ["Bench","Bench (No SS)", "No Risk", "No Risk (No SS)",
  "Exog. Labor", "Exog. Labor (No SS)"]

K_vals = hcat(round(baseline[1],3), round(baseline_no_ss[1],3),
  round(no_idio_risk[1],3), round(no_idio_risk_no_ss[1],3), round(exo_labor[1],3),
  round(exo_labor_no_ss[1],3))
L_vals = hcat(round(baseline[2],3), round(baseline_no_ss[2],3),
  round(no_idio_risk[2],3), round(no_idio_risk_no_ss[2],3), round(exo_labor[2],3),
  round(exo_labor_no_ss[2],3))
w_vals = hcat(round(baseline[3],3), round(baseline_no_ss[3],3),
  round(no_idio_risk[3],3), round(no_idio_risk_no_ss[3],3), round(exo_labor[3],3),
  round(exo_labor_no_ss[3],3))
r_vals = hcat(round(baseline[4],3), round(baseline_no_ss[4],3),
  round(no_idio_risk[4],3), round(no_idio_risk_no_ss[4],3), round(exo_labor[4],3),
  round(exo_labor_no_ss[4],3))
b_vals = hcat(round(baseline[5],3), round(baseline_no_ss[5],3),
  round(no_idio_risk[5],3), round(no_idio_risk_no_ss[5],3), round(exo_labor[5],3),
  round(exo_labor_no_ss[5],3))
W_vals = hcat(round(baseline[6],3), round(baseline_no_ss[6],3),
  round(no_idio_risk[6],3), round(no_idio_risk_no_ss[6],3), round(exo_labor[6],3),
  round(exo_labor_no_ss[6],3))
cv_vals = hcat(round(baseline[7],3), round(baseline_no_ss[7],3),
  round(no_idio_risk[7],3), round(no_idio_risk_no_ss[7],3), round(exo_labor[7],3),
  round(exo_labor_no_ss[7],3))
vote_vals = hcat(round(baseline_vote_yes,3),string(" "),
  round(no_idio_risk_vote_yes,3), string(" "), round(exo_labor_vote_yes,3),
  string(" "))

output[2:9,1] = titles_vert
output[1,2:7] = titles_horz
output[2,2:7] = K_vals
output[3,2:7] = L_vals
output[4,2:7] = w_vals
output[5,2:7] = r_vals
output[6,2:7] = b_vals
output[7,2:7] = W_vals
output[8,2:7] = cv_vals
output[1,1] = string(" ")
output[9,2:7] = vote_vals

tabular(output)

#= Graphs =#

v50fig = figure()
plot(prim.a_vals,v_50,color="blue",linewidth=2.0)
xlabel("a")
ylabel("value")
legend(loc="lower right")
title("Value Function (Age 50)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS5/Pictures/v50.pgf")

policy20fig = figure()
plot(prim.a_vals,prim.a_vals[policy_20[:,1]],color="blue",linewidth=2.0,label="High Productivity")
plot(prim.a_vals,prim.a_vals[policy_20[:,2]],color="red",linewidth=2.0,linestyle="--",label="Low Productivity")
plot(prim.a_vals,prim.a_vals,color="yellow",linewidth=1.0,label="45 Degree")
xlabel("a")
ylabel("a'(a,z)")
legend(loc="lower right")
title("Policy Function (Age 20)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS5/Pictures/policy20.pgf")
