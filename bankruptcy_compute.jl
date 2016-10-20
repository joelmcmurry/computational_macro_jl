#=
Program Name: bankruptcy_compute.jl
Runs bankruptcy model
=#

using PyPlot

include("bankruptcy_model.jl")

function compute_pooling(;q0=0.01,max_iter=100,
  max_iter_vfi=2000,epsilon=1e-3,a_size=500)

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
    results = SolveProgram(prim,v0,v1,bellman_clean_pool!,max_iter_vfi=max_iter_vfi)

    ## Calculate loss rate for lenders
      # Total assets lent. Note only lent to no-bankrupt history agents
      L = 0.0
      for state_index in 1:prim.s_size
        for asset_index in 1:prim.a_size
          choice_index = results.sigma0[asset_index,state_index]
          if choice_index < prim.zero_index && choice_index != 0
            dist_index = asset_index+(state_index-1)*a_size # pick out entry in stationary distribution
            L += -prim.a_vals[choice_index]*results.statdist[dist_index]
          end
        end
      end

      # Total defaulted assets
      D = 0.0
      for state_index in 1:prim.s_size
        for asset_index in 1:prim.a_size
          choice_index = results.sigma0[asset_index,state_index]
          if choice_index == 0
            dist_index = asset_index+(state_index-1)*a_size # pick out entry in stationary distribution
            D += -prim.a_vals[asset_index]*results.statdist[dist_index]
          end
        end
      end

      # Loss rate
      Deltaprime = D/L

      profits_pool = (1 - Deltaprime)/(1 + prim.r) - q_pool

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Profits: ", profits_pool," q_pool: ", q_pool)

    # Adjust q (and stop if asset market clears)
    if abs(profits_pool) < epsilon
        break
    elseif profits_pool > 0 # q too small
      qlower = q_pool
    else # q too big
      qupper = q_pool
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

#= Since policy functions are not defined over default areas and
value functions are not defined over negative assets for bankrupt
histories, need to trim arrays =#

policy_pool0 = pooling_results.sigma0[:,1]

#policy_emp0_pool = pooling_prim.a_vals[pooling_results.sigma0[:,1]]
#policy_emp1_pool = pooling_prim.a_vals[pooling_results.sigma1[:,1]]
#policy_unemp0_pool = pooling_prim.a_vals[pooling_results.sigma0[:,2]]
#policy_unemp1_pool = pooling_prim.a_vals[pooling_results.sigma1[:,2]]
value_emp0_pool = pooling_results.Tv0[:,1]
value_emp1_pool = pooling_results.Tv1[:,1]
value_unemp0_pool = pooling_results.Tv0[:,2]
value_unemp1_pool = pooling_results.Tv1[:,2]

# Plot value functions

valfig = figure()
plot(pooling_prim.a_vals,value_emp0_pool,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(pooling_prim.a_vals,value_unemp0_pool,color="red",linewidth=2.0,label="Unemployed (h=0)")
xlabel("a")
ylabel("v(a,s,h)")
legend(loc="lower right")
title("Value Functions")
ax = PyPlot.gca()
ax[:set_ylim]((-20,1))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4/Pictures/policyfunctions.pgf")

# Plot value function

valfig = figure()
plot(huggett.a_vals,value_emp,color="blue",linewidth=2.0,label="Employed")
plot(huggett.a_vals,value_unemp,color="red",linewidth=2.0,label="Unemployed")
xlabel("a")
ylabel("V(a,s)")
legend(loc="lower right")
title("Value Functions")
ax = PyPlot.gca()
ax[:set_ylim]((-10,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4/Pictures/valuefunctions.pgf")


# Plot stationary distribution

distfig = figure()
bar(huggett.a_vals,huggett_results.statdist[1:huggett.a_size],
  color="blue",label="Employed")
bar(huggett.a_vals,huggett_results.statdist[huggett.a_size+1:huggett.N],
  color="red",label="Unemployed")
title("Wealth Distribution")
legend(loc="upper right")
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4/Pictures/stationarydistributions.pgf")
