#=
Program Name: conesa_krueger_compute.jl
Runs Conesa-Krueger model and performs policy experiments
=#

using PyPlot
using LatexPrint

include("conesa_krueger_model.jl")

#= Solve model for example years =#

# Initialize Primitives
prim = Primitives(a_size=1000)

#= Solve worker problem and return value functions and policy functions
for 50-year old and a 20-year old =#
tic()
res, v_20, v_50, policy_20, policy_50, labor_supply_20 =
  SolveProgram(prim,20,50)
toc()

#= Policy Experiments =#

function policy_experiment(;K0_ss::Float64=2.0,L0_ss::Float64=0.3,
    K0_no_ss::Float64=2.0,L0_no_ss::Float64=0.3,z_vals::Array{Float64}=[3.0,0.5],
    gamma::Float64=0.42,a_size=1000,max_iter=100)
  # with social security
  with_ss = compute_GE(a_size=a_size,max_iter=max_iter,K0=K0_ss,L0=L0_ss,
  gamma=gamma,z_vals=z_vals)
  # without social security
  without_ss = compute_GE(a_size=a_size,max_iter=max_iter,K0=K0_no_ss,
    L0=L0_no_ss,gamma=gamma,z_vals=z_vals,theta=0.00)

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

# Baseline: idiosynractic risk and endogenous labor

tic()
baseline, baseline_no_ss, baseline_vote_yes = policy_experiment(K0_ss=1.99,
  L0_ss=0.32,K0_no_ss=2.5,L0_no_ss=0.34,gamma=0.42,z_vals=[3.0,0.5])
toc()

# No idiosyncratic risk

tic()
no_idio_risk, no_idio_risk_no_ss, no_idio_risk_vote_yes =
  policy_experiment(K0_ss=1.99,L0_ss=0.32,K0_no_ss=2.5,L0_no_ss=0.34,
  gamma=0.42,z_vals=[0.5,0.5])
toc()

# Exogenous Labor

tic()
exo_labor, exo_labor_no_ss, exo_labor_vote_yes =
  policy_experiment(K0_ss=4.7,L0_ss=0.75,K0_no_ss=6.0,L0_no_ss=0.75,
  gamma=1.0,z_vals=[3.0,0.5])
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

policy20fig = figure()
plot(prim.a_vals,prim.a_vals[policy_20[:,1]],color="blue",linewidth=2.0,label="High Productivity")
plot(prim.a_vals,prim.a_vals[policy_20[:,2]],color="red",linewidth=2.0,linestyle="--",label="Low Productivity")
plot(prim.a_vals,prim.a_vals,color="yellow",linewidth=1.0,label="45 Degree")
xlabel("a")
ylabel("a'(a,z)")
legend(loc="lower right")
title("Policy Function (Age 20)")
ax = PyPlot.gca()
