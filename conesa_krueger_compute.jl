#=
Program Name: conesa_krueger_compute.jl
Runs Conesa-Krueger model
=#

using PyPlot

include("conesa_krueger_model.jl")

## Initialize Primitives

prim = Primitives(a_size=1000,a_max=100.00)

#= Solve worker problem and return value functions and policy functions
for 50-year old and a 20-year old =#
tic()
res, v_20, v_50, policy_20, policy_50, labor_supply_20 = SolveProgram(prim)
toc()

#= General Equilibrium =#

function compute_GE(;a_size=100,epsilon=1e-4,max_iter=100,
    K0::Float64=2.0363,L0::Float64=0.3249)

  # Initialize primitives
  prim = Primitives(a_size=a_size)

  # Solve problem with default values
  results = SolveProgram(prim)[1]

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

    results = SolveProgram(prim)[1]

    # Calculate new aggregate capital and labor

    K_new = 0.00
    for working_age in 1:prim.JR-1
      K_new += dot(results.ss_working_hi[:,working_age],
        prim.a_vals[results.policy_working_hi[:,working_age]])
      K_new += dot(results.ss_working_lo[:,working_age],
        prim.a_vals[results.policy_working_lo[:,working_age]])
    end
    for retired_age in 1:prim.N-prim.JR+1
      K_new += dot(results.ss_retired[:,retired_age],
        prim.a_vals[results.policy_retired[:,retired_age]])
    end

    L_new = 0.00
    for working_age in 1:prim.JR-1
      L_new += dot(results.ss_working_hi[:,working_age],
        results.labor_supply_hi[:,working_age]*prim.ageeff[working_age])
      L_new += dot(results.ss_working_lo[:,working_age],
        results.labor_supply_lo[:,working_age]*prim.ageeff[working_age])
    end

    # Adjust K, L if fails tolerance
    max_dist = max(abs(K-K_new),abs(L-L_new))
    if max_dist < epsilon
        break
    else
      L_new = L*0.99 + L_new*0.01
      K_new = K*0.99 + K_new*0.01
    end

    # Calculate new prices
    w_new = (1-prim.alpha)*K_new^(prim.alpha)*L_new^(-prim.alpha)
    r_new = prim.alpha*K_new^(prim.alpha-1)*L_new^(1-prim.alpha) - prim.delta

    prim.w = w_new
    prim.r = r_new
    L = L_new
    K = K_new

  end

end
tic()
compute_GE(a_size=1000,max_iter=100)
toc()

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
