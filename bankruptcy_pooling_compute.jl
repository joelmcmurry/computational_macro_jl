#=
Program Name: bankruptcy_pooling_compute.jl
Runs bankruptcy model (pooling)
=#

using PyPlot

include("bankruptcy_model.jl")

#= Pooling Equilibrium =#

function compute_pooling(;q0=0.01,max_iter=100,epsilon=1e-4,
  max_iter_vfi=2000,max_iter_statdist=500,
  epsilon_vfi=1e-4,epsilon_statdist=1e-4,
  a_size=500)

  # Starting range for pooling discount bond price
  qlower = q0
  qupper = 1

  q_pool = (qlower + qupper)/2

  # Initialize primitives
  prim = Primitives(q_pool=q_pool,a_size=a_size)

  # Initial guess for value functions
  v0 = zeros(prim.a_size,prim.s_size)
  v1 = zeros(prim.a_size,prim.s_size)

  # Initialize lender profits
  profits_pool = 10.0

  # Initialize results structure
  results = Results(prim,v0,v1)

  for i in 1:max_iter

    # Solve dynamic program given new q_pool
    results = SolveProgram(prim,v0,v1,bellman_clean_pool!,
      max_iter_vfi=max_iter_vfi,max_iter_statdist=max_iter_statdist,
      epsilon_vfi=epsilon_vfi,epsilon_statdist=epsilon_statdist,
      distflag="yes")

    ## Calculate loss rate for lenders
      # Total assets lent. Note only lent to no-bankrupt history agents
      L = 0.0
      for state_index in 1:prim.s_size
        for asset_index in 1:prim.a_size
          choice_index = results.sigma0[asset_index,state_index]
          if choice_index < prim.zero_index && results.d0[asset_index,state_index] != 1
            dist_index = asset_index+(state_index-1)*prim.a_size # pick out entry in stationary distribution
            L += -prim.a_vals[choice_index]*results.statdist[dist_index]
          end
        end
      end

      # Total defaulted assets
      D = 0.0
      for state_index in 1:prim.s_size
        for asset_index in 1:prim.a_size
          if results.d0[asset_index,state_index] == 1
            # pick out entry in stationary distribution
            dist_index = asset_index+(state_index-1)*prim.a_size
            D += -prim.a_vals[asset_index]*results.statdist[dist_index]
          end
        end
      end

      # Loss rate
      Deltaprime = D/L

      profits_pool = prim.q_pool - (1 - Deltaprime)/(1 + prim.r)

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Profits: ", profits_pool," q_pool: ", prim.q_pool)

    # Adjust q (and stop if asset market clears)
    if abs(profits_pool) < epsilon
        break
    elseif profits_pool > 0.00 # q too large
      qupper = prim.q_pool
    else # q too small
      qlower = prim.q_pool
    end

    q_pool = (qlower + qupper)/2

    # Update primitives given new q
    prim.q_pool = q_pool

    # Update guess for value function
    v0 = results.Tv0
    v1 = results.Tv1

  end

profits_pool, q_pool, prim, results

end

tic()
pooling_eq = compute_pooling(max_iter=100,a_size=500)
toc()

profits_pool = pooling_eq[1]
q_pool = pooling_eq[2]
prim_pool = pooling_eq[3]
results_pool = pooling_eq[4]

#= Collapse stationary distribution over all state/histories
to stationary distribution over assets. For each asset level there
are four possible (s,h) combinations =#

statdist_assets_pool = results_pool.statdist[1:prim_pool.a_size] +
  results_pool.statdist[prim_pool.a_size+1:prim_pool.N] +
  results_pool.statdist[prim_pool.N+1:prim_pool.N+prim_pool.a_size] +
  results_pool.statdist[prim_pool.N+prim_pool.a_size+1:2*prim_pool.N]

#= Output Moments =#

  # Income

  avg_inc_pool = sum(prim_pool.s_vals[1]*(results_pool.statdist[1:prim_pool.a_size]+
    results_pool.statdist[prim_pool.N+1:prim_pool.N+prim_pool.a_size])) +
    sum(prim_pool.s_vals[2]*(results_pool.statdist[prim_pool.a_size+1:prim_pool.N]+
      results_pool.statdist[prim_pool.N+prim_pool.a_size+1:2*prim_pool.N]))

  # Savings

  avg_savings_pool = dot(statdist_assets_pool[prim_pool.zero_index:prim_pool.a_size],
    prim_pool.a_vals[prim_pool.zero_index:prim_pool.a_size])

  # Debt

  avg_debt_pool = dot(statdist_assets_pool[1:prim_pool.zero_index],
    prim_pool.a_vals[1:prim_pool.zero_index])

  # Default amount

  default_debt_pool = dot(prim_pool.a_vals,
    results_pool.statdist[prim_pool.a_size+1:prim_pool.N].*results_pool.d0[:,2])

  # Default rate (fraction of debt holders that default)

  default_rate_pool = dot(results_pool.statdist[1:prim_pool.zero_index],
    results_pool.d0[1:prim_pool.zero_index,1]) +
    dot(results_pool.statdist[prim_pool.a_size+1:prim_pool.a_size+prim_pool.zero_index],
    results_pool.d0[1:prim_pool.zero_index,2])

  # Debt-to-Income

  debt_to_income_pool = avg_debt_pool/avg_inc_pool

  summary_pool = [debt_to_income_pool, avg_inc_pool, avg_savings_pool, avg_debt_pool,
    default_rate_pool, default_debt_pool, q_pool]

## Consumption Equivalent

# Generate separating equilibrium
include("bankruptcy_separating_compute.jl")

  # Calculate consumption equivalent (pooling to separating) for each history

  if prim_pool.a_size == prim_sep.a_size

    lambda0 = ((results_sep.Tv0 + (1/((1-prim_pool.alpha)*(1-prim_pool.beta)))).^(1/(1-prim_pool.alpha))).*
      ((results_pool.Tv0 .+ (1/((1-prim_pool.alpha)*(1-prim_pool.beta)))).^(1/(prim_pool.alpha-1))).-1

    lambda1 = ((results_sep.Tv1 + (1/((1-prim_pool.alpha)*(1-prim_pool.beta)))).^(1/(1-prim_pool.alpha))).*
      ((results_pool.Tv1 .+ (1/((1-prim_pool.alpha)*(1-prim_pool.beta)))).^(1/(prim_pool.alpha-1))).-1

  else
    msg = "Asset grid sizes must match between pooling and separating models"
    throw(ArgumentError(msg))
  end

#= Graphs =#

#= Since policy functions are not defined over default regions
need to trim index arrays and only return non-default policies. Also need to
construct matching asset values for plotting =#

policyindex_emp1_pool = results_pool.sigma1[:,1][results_pool.sigma1[:,1].!=0]
  a_vals_policy_emp1_pool = prim_pool.a_vals[results_pool.sigma1[:,1].!=0]
policyindex_unemp1_pool = results_pool.sigma1[:,2][results_pool.sigma1[:,2].!=0]
  a_vals_policy_unemp1_pool = prim_pool.a_vals[results_pool.sigma1[:,2].!=0]

# Return values for policy functions and no-bankrupt value functions

policy_emp0_pool = prim_pool.a_vals[results_pool.sigma0[:,1]]
policy_unemp0_pool = prim_pool.a_vals[results_pool.sigma0[:,2]]
policy_emp1_pool = prim_pool.a_vals[policyindex_emp1_pool]
policy_unemp1_pool = prim_pool.a_vals[policyindex_unemp1_pool]
value_emp0_pool = results_pool.Tv0[:,1]
value_unemp0_pool = results_pool.Tv0[:,2]

#= Value functions only defined for bankrupt histories over positive assets.
Trim value function arrays and construct corresponding asset values =#

value_emp1_pool = results_pool.Tv1[:,1][results_pool.Tv1[:,1].!=-Inf]
  a_vals_value_emp1_pool = prim_pool.a_vals[results_pool.Tv1[:,1].!=-Inf]
value_unemp1_pool = results_pool.Tv1[:,2][results_pool.Tv1[:,2].!=-Inf]
  a_vals_value_unemp1_pool = prim_pool.a_vals[results_pool.Tv1[:,2].!=-Inf]

# Plot value functions

valfig0 = figure()
plot(prim_pool.a_vals,value_emp0_pool,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(prim_pool.a_vals,value_unemp0_pool,color="red",linewidth=2.0,label="Unemployed (h=0)")
xlabel("a")
ylabel("v(a,s,h)")
legend(loc="lower right")
title("Value Functions (Pooling - No Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-20,2))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/valuefunctions0_pool.pgf")

valfig1 = figure()
plot(a_vals_value_emp1_pool,value_emp1_pool,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_value_unemp1_pool,value_unemp1_pool,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
xlabel("a")
ylabel("v(a,s,h)")
legend(loc="lower right")
title("Value Functions (Pooling - Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-20,2))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/valuefunctions1_pool.pgf")

# Plot value function

polfig0 = figure()
plot(prim_pool.a_vals,policy_emp0_pool,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(prim_pool.a_vals,policy_unemp0_pool,color="red",linewidth=2.0,label="Unemployed (h=0)")
plot(prim_pool.a_vals,prim_pool.a_vals,color="black",linewidth=1.0)
xlabel("a")
ylabel("g(a,s,h)")
legend(loc="lower right")
title("Policy Functions (Pooling - No Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,5))
ax[:set_xlim]((-0.525,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/policyfunctions0_pool.pgf")

polfig1 = figure()
plot(a_vals_policy_emp1_pool,policy_emp1_pool,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_policy_unemp1_pool,policy_unemp1_pool,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
plot(a_vals_policy_emp1_pool,a_vals_policy_emp1_pool,color="black",linewidth=1.0)
xlabel("a")
ylabel("g(a,s,h)")
legend(loc="lower right")
title("Policy Functions (Pooling - Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,5))
ax[:set_xlim]((0,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/policyfunctions1_pool.pgf")

# Plot default decision rule

decfig = figure()
plot(prim_pool.a_vals,results_pool.d0[:,1],color="blue",linewidth=2.0,label="Employed")
plot(prim_pool.a_vals,results_pool.d0[:,2],color="red",linewidth=2.0,label="Unemployed")
xlabel("a")
ylabel("d(a,s,0)")
legend(loc="lower right")
title("Default Decision Rule (Pooling)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,2))
ax[:set_xlim]((-0.525,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/decisionrule_pool.pgf")

# Bond prices

bondfig = figure()
plot(prim_pool.a_vals,prim_pool.q_pool*ones(Float64,prim_pool.a_size),color="blue",linewidth=2.0,label="Employed")
plot(prim_pool.a_vals,prim_pool.q_pool*ones(Float64,prim_pool.a_size),color="red",linewidth=2.0,label="Unemployed")
xlabel("a")
ylabel("q(a,s,0)")
legend(loc="lower right")
title("Bond Prices (Pooling)")
ax = PyPlot.gca()
ax[:set_ylim]((0,1))
ax[:set_xlim]((-0.525,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/bondprices_pool.pgf")

# Distribution

distfig1 = figure()
PyPlot.bar(prim_pool.a_vals,results_pool.statdist[1:prim_pool.a_size],
  width=0.1,alpha=0.5,color="blue",label="mu(a,s=1,h=0)")
title("Distribution - Pooling (Employed/No-Bankruptcy)")
legend(loc="upper right")
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/distemp0_pool.pgf")

distfig2 = figure()
PyPlot.bar(prim_pool.a_vals,results_pool.statdist[prim_pool.a_size+1:prim_pool.N],
  width=0.1,alpha=0.5,color="red",label="mu(a,s=0.05,h=0)")
title("Distribution - Pooling (Unemployed/No-Bankruptcy)")
legend(loc="upper right")
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/distunemp0_pool.pgf")

distfig3 = figure()
PyPlot.bar(prim_pool.a_vals,results_pool.statdist[prim_pool.N+1:prim_pool.N+prim_pool.a_size],
  width=0.1,alpha=0.5,color="green",label="mu(a,s=1,h=1)")
title("Distribution - Pooling (Employed/Bankruptcy)")
legend(loc="upper right")
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/distemp1_pool.pgf")

distfig4 = figure()
PyPlot.bar(prim_pool.a_vals,results_pool.statdist[prim_pool.N+prim_pool.a_size+1:prim_pool.N*2],width=0.1,alpha=0.5,color="yellow",label="mu(a,s=0.05,h=1)")
title("Distribution - Pooling (Unemployed/Bankruptcy)")
legend(loc="upper right")
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/distunemp1_pool.pgf")

# Consumption Equivalents

conseq0 = figure()
plot(prim_pool.a_vals,lambda0[:,1],color="blue",linewidth=2.0,label="Employed (h=0)")
plot(prim_pool.a_vals,lambda0[:,2],color="red",linewidth=2.0,label="Unemployed (h=0)")
xlabel("a")
legend(loc="lower right")
title("Consumption Equivalents (h=0)")
ax = PyPlot.gca()
ax[:set_ylim]((0,0.1))
ax[:set_xlim]((-0.525,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/consequiv0.pgf")

conseq1 = figure()
plot(prim_pool.a_vals,lambda1[:,1],color="green",linewidth=2.0,label="Employed (h=1)")
plot(prim_pool.a_vals,lambda1[:,2],color="yellow",linewidth=2.0,label="Unemployed (h=1)")
xlabel("a")
legend(loc="lower right")
title("Consumption Equivalents (h=1)")
ax = PyPlot.gca()
ax[:set_ylim]((0,0.02))
ax[:set_xlim]((-0.525,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/consequiv1.pgf")
