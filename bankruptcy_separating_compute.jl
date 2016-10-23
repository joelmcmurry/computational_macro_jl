#=
Program Name: bankruptcy_separating_compute.jl
Runs bankruptcy model (separating)
=#

using PyPlot

include("bankruptcy_model.jl")

#= Separating Equilibrium =#

function compute_separating(;q0=0.01,max_iter=100,epsilon=1e-4,
  max_iter_vfi=2000,max_iter_statdist=500,
  epsilon_vfi=1e-4,epsilon_statdist=1e-4,
  a_size=500)

  ## Starting range for pooling discount bond price

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
      epsilon_vfi=epsilon_vfi,epsilon_statdist=epsilon_statdist)

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

      profits_pool = (1 - Deltaprime)/(1 + prim.r) - prim.q_pool

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Profits: ", profits_pool," q_pool: ", prim.q_pool)

    # Adjust q (and stop if asset market clears)
    if abs(profits_pool) < epsilon
        break
    elseif profits_pool > 0.00 # q too small
      qlower = prim.q_pool
    else # q too big
      qupper = prim.q_pool
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
results = compute_pooling(max_iter=100,a_size=2000)
toc()

pooling_prim = results[3]
pooling_results = results[4]

#= Since policy functions are not defined over default regions
need to trim index arrays and only return non-default policies. Also need to
construct matching asset values for plotting =#

policyindex_emp1_pool = pooling_results.sigma1[:,1][pooling_results.sigma1[:,1].!=0]
  a_vals_policy_emp1_pool = pooling_prim.a_vals[pooling_results.sigma1[:,1].!=0]
policyindex_unemp1_pool = pooling_results.sigma1[:,2][pooling_results.sigma1[:,2].!=0]
  a_vals_policy_unemp1_pool = pooling_prim.a_vals[pooling_results.sigma1[:,2].!=0]

# Return values for policy functions and no-bankrupt value functions

policy_emp0_pool = pooling_prim.a_vals[pooling_results.sigma0[:,1]]
policy_unemp0_pool = pooling_prim.a_vals[pooling_results.sigma0[:,2]]
policy_emp1_pool = pooling_prim.a_vals[policyindex_emp1_pool]
policy_unemp1_pool = pooling_prim.a_vals[policyindex_unemp1_pool]
value_emp0_pool = pooling_results.Tv0[:,1]
value_unemp0_pool = pooling_results.Tv0[:,2]

#= Value functions only defined for bankrupt histories over positive assets.
Trim value function arrays and construct corresponding asset values =#

value_emp1_pool = pooling_results.Tv1[:,1][pooling_results.Tv1[:,1].!=-Inf]
  a_vals_value_emp1_pool = pooling_prim.a_vals[pooling_results.Tv1[:,1].!=-Inf]
value_unemp1_pool = pooling_results.Tv1[:,2][pooling_results.Tv1[:,2].!=-Inf]
  a_vals_value_unemp1_pool = pooling_prim.a_vals[pooling_results.Tv1[:,2].!=-Inf]

# Plot value functions

valfig = figure()
plot(pooling_prim.a_vals,value_emp0_pool,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(pooling_prim.a_vals,value_unemp0_pool,color="red",linewidth=2.0,label="Unemployed (h=0)")
plot(a_vals_value_emp1_pool,value_emp1_pool,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_value_unemp1_pool,value_unemp1_pool,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
xlabel("a")
ylabel("v(a,s,h)")
legend(loc="lower right")
title("Value Functions (Pooling)")
ax = PyPlot.gca()
ax[:set_ylim]((-20,2))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/valuefunctions_pool.pgf")

# Plot value function

polfig = figure()
plot(pooling_prim.a_vals,policy_emp0_pool,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(pooling_prim.a_vals,policy_unemp0_pool,color="red",linewidth=2.0,label="Unemployed (h=0)")
plot(a_vals_policy_emp1_pool,policy_emp1_pool,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_policy_unemp1_pool,policy_unemp1_pool,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
plot(pooling_prim.a_vals,pooling_prim.a_vals,color="black",linewidth=1.0)
xlabel("a")
ylabel("g(a,s,h)")
legend(loc="lower right")
title("Policy Functions (Pooling)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,5))
ax[:set_xlim]((-0.525,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/policyfunctions_pool.pgf")
