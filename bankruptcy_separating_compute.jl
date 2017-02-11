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

  # Initialize primitives
  prim = Primitives(a_size=a_size)

  # Starting range for pooling discount bond price
  qlower = q0*ones(Float64,prim.a_size,prim.s_size)
  qupper = ones(Float64,prim.a_size,prim.s_size)

  q_menu = (qlower+qupper)./2
  prim.q_menu = q_menu

  # Initial guess for value functions
  v0 = zeros(prim.a_size,prim.s_size)
  v1 = zeros(prim.a_size,prim.s_size)

  # Initialize lender profits
  profits_sep = ones(Float64,prim.a_size,prim.s_size)*10.0

  # Initialize results structure
  results = Results(prim,v0,v1)

  for i in 1:max_iter

    # Solve dynamic program given new q_sep menu
    results = SolveProgram(prim,v0,v1,bellman_clean_sep!,
      max_iter_vfi=max_iter_vfi,epsilon_vfi=epsilon_vfi,
      distflag="no")

    ## Calculate loss rate for lenders (on each contract)

    # Initialize lender loss rate
    delta_sep = zeros(Float64,prim.a_size,prim.s_size)

    for state_today in 1:prim.s_size
      for asset_tomorrow in 1:prim.a_size
        if results.d0[asset_tomorrow,1] == 1 # look in high state tomorrow
          delta_sep[asset_tomorrow,state_today] += prim.markov[state_today,1]
        end
        if results.d0[asset_tomorrow,2] == 1 # look in low state tomorrow
          delta_sep[asset_tomorrow,state_today] += prim.markov[state_today,2]
        end
      end
    end

    zero_profit_target = 1/(1+prim.r)*ones(Float64,prim.a_size,prim.s_size) - delta_sep./(1+prim.r)

    # Calculate implicit borrowing constraints
    for state_today in 1:prim.s_size
      if isempty(find(zero_profit_target[:,state_today].==0.0)) == false
        prim.a_min_sep[state_today] =
          prim.a_vals[maximum(find(zero_profit_target[:,state_today].==0.0))]
      else
        prim.a_min_sep[state_today] =
          prim.a_vals[minimum(find(zero_profit_target[:,state_today].>0.0))]
      end
    end

    profits_sep = prim.q_menu - zero_profit_target

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Max Profits: ", maximum(profits_sep),
    " Min Profits: ", minimum(profits_sep))

    if maximum(abs(profits_sep)) < epsilon
        break
    else
      q_menu = (prim.q_menu+zero_profit_target)./2
    end


    # Update primitives given new q menu
    #q_menu = (qlower + qupper)./2
    prim.q_menu = q_menu

    # Update guess for value function
    v0 = results.Tv0
    v1 = results.Tv1

  end

  ## Find stationary distribution

  create_statdist!(prim,results,max_iter_statdist,
    epsilon_statdist)

profits_sep, q_menu, prim, results

end

tic()
separating_eq = compute_separating(max_iter=100,a_size=500)
toc()

profits_sep = separating_eq[1]
q_menu = separating_eq[2]
prim_sep = separating_eq[3]
results_sep = separating_eq[4]

#= Collapse stationary distribution over all state/histories
to stationary distribution over assets. For each asset level there
are four possible (s,h) combinations =#

statdist_assets_sep = results_sep.statdist[1:prim_sep.a_size] +
  results_sep.statdist[prim_sep.a_size+1:prim_sep.N] +
  results_sep.statdist[prim_sep.N+1:prim_sep.N+prim_sep.a_size] +
  results_sep.statdist[prim_sep.N+prim_sep.a_size+1:2*prim_sep.N]

#= Output Moments =#

  # Income

  avg_inc_sep = sum(prim_sep.s_vals[1]*(results_sep.statdist[1:prim_sep.a_size]+
    results_sep.statdist[prim_sep.N+1:prim_sep.N+prim_sep.a_size])) +
    sum(prim_sep.s_vals[2]*(results_sep.statdist[prim_sep.a_size+1:prim_sep.N]+
      results_sep.statdist[prim_sep.N+prim_sep.a_size+1:2*prim_sep.N]))

  # Savings

  avg_savings_sep = dot(statdist_assets_sep[prim_sep.zero_index:prim_sep.a_size],
    prim_sep.a_vals[prim_sep.zero_index:prim_sep.a_size])

  # Debt

  avg_debt_sep = dot(statdist_assets_sep[1:prim_sep.zero_index],
    prim_sep.a_vals[1:prim_sep.zero_index])

  # Default amount

  default_debt_sep = dot(prim_sep.a_vals,
    results_sep.statdist[prim_sep.a_size+1:prim_sep.N].*results_sep.d0[:,2])

  # Default rate (fraction of debt holders that default)

  default_rate_sep = dot(results_sep.statdist[1:prim_sep.zero_index],
    results_sep.d0[1:prim_sep.zero_index,1]) +
    dot(results_sep.statdist[prim_sep.a_size+1:prim_sep.a_size+prim_sep.zero_index],
    results_sep.d0[1:prim_sep.zero_index,2])

  # Average bond price (only for borrowers)

  weighted_sum_borrow = 0.0
  mass_borrow = 0.0

  for state_today in 1:prim_sep.s_size
    for asset_today in 1:prim_sep.a_size
      if results_sep.sigma0[asset_today,state_today] < prim_sep.zero_index
        weighted_sum_borrow += q_menu[results_sep.sigma0[asset_today,state_today],
          state_today]*results_sep.statdist[asset_today+(state_today-1)*
          prim_sep.a_size]
        mass_borrow += results_sep.statdist[asset_today+(state_today-1)*
        prim_sep.a_size]
      end
    end
  end
  avg_q_sep = weighted_sum_borrow/mass_borrow

  # Debt-to-Income

  debt_to_income_sep = avg_debt_sep/avg_inc_sep

  summary_sep = [debt_to_income_sep, avg_inc_sep, avg_savings_sep, avg_debt_sep,
    default_rate_sep, default_debt_sep, avg_q_sep]

## Calculate interest rate menu implied by bond prices

  int_menu_sep = ones(Float64,prim_sep.a_size,prim_sep.s_size)./
    prim_sep.q_menu-ones(Float64,prim_sep.a_size,prim_sep.s_size)

  # Find interest rates chosen in equilibrium

  int_chosen_sep = zeros(Float64,prim_sep.a_size,prim_sep.s_size)

  for state_today in 1:prim_sep.s_size
    for asset_today in 1:prim_sep.a_size
      int_chosen_sep[asset_today,state_today] =
          int_menu_sep[results_sep.sigma0[asset_today,state_today],
            state_today]
    end
  end

  # Calculate distribution of interest rates chosen in equilibrium

  int_rate_emp = sort(union(int_chosen_sep[:,1],linspace(0,1,100)))
  int_rate_unemp = sort(union(int_chosen_sep[:,2],linspace(0,1,100)))

  int_dist_emp = zeros(Float64,size(int_rate_emp))
  int_dist_unemp = zeros(Float64,size(int_rate_unemp))

  # Distribution for employed (normalize by mass of employed/no-bankrupt)
  for i in 1:size(int_rate_emp)[1]
    for j in 1:prim_sep.a_size
      if int_chosen_sep[j,1] == int_rate_emp[i]
        int_dist_emp[i] += results_sep.statdist[j]
      end
    end
  end
  int_dist_emp=int_dist_emp./sum(results_sep.statdist[1:prim_sep.a_size])

  # Distribution for unemployed (normalize by mass of unemployed/no-bankrupt)
  for i in 1:size(int_rate_unemp)[1]
    for j in 1:prim_sep.a_size
      if int_chosen_sep[j,2] == int_rate_unemp[i]
        int_dist_unemp[i] += results_sep.statdist[j+prim_sep.a_size]
      end
    end
  end
  int_dist_unemp=int_dist_unemp./sum(results_sep.statdist[prim_sep.a_size+1:prim_sep.N])

#= Graphs =#

#= Since policy functions are not defined over default regions
need to trim index arrays and only return non-default policies. Also need to
construct matching asset values for plotting =#

policyindex_emp1_sep = results_sep.sigma1[:,1][results_sep.sigma1[:,1].!=0]
  a_vals_policy_emp1_sep = prim_sep.a_vals[results_sep.sigma1[:,1].!=0]
policyindex_unemp1_sep = results_sep.sigma1[:,2][results_sep.sigma1[:,2].!=0]
  a_vals_policy_unemp1_sep = prim_sep.a_vals[results_sep.sigma1[:,2].!=0]

# Return values for policy functions and no-bankrupt value functions

policy_emp0_sep = prim_sep.a_vals[results_sep.sigma0[:,1]]
policy_unemp0_sep = prim_sep.a_vals[results_sep.sigma0[:,2]]
policy_emp1_sep = prim_sep.a_vals[policyindex_emp1_sep]
policy_unemp1_sep = prim_sep.a_vals[policyindex_unemp1_sep]
value_emp0_sep = results_sep.Tv0[:,1]
value_unemp0_sep = results_sep.Tv0[:,2]

#= Value functions only defined for bankrupt histories over positive assets.
Trim value function arrays and construct corresponding asset values =#

value_emp1_sep = results_sep.Tv1[:,1][results_sep.Tv1[:,1].!=-Inf]
  a_vals_value_emp1_sep = prim_sep.a_vals[results_sep.Tv1[:,1].!=-Inf]
value_unemp1_sep = results_sep.Tv1[:,2][results_sep.Tv1[:,2].!=-Inf]
  a_vals_value_unemp1_sep = prim_sep.a_vals[results_sep.Tv1[:,2].!=-Inf]

# Plot value functions

valfig0 = figure()
plot(prim_sep.a_vals,value_emp0_sep,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(prim_sep.a_vals,value_unemp0_sep,color="red",linewidth=2.0,label="Unemployed (h=0)")
xlabel("a")
ylabel("v(a,s,h)")
legend(loc="lower right")
title("Value Functions (Separating - No Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-20,2))

valfig1 = figure()
plot(a_vals_value_emp1_sep,value_emp1_sep,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_value_unemp1_sep,value_unemp1_sep,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
xlabel("a")
ylabel("v(a,s,h)")
legend(loc="lower right")
title("Value Functions (Separating - Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-20,2))

# Plot value function

polfig0 = figure()
plot(prim_sep.a_vals,policy_emp0_sep,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(prim_sep.a_vals,policy_unemp0_sep,color="red",linewidth=2.0,label="Unemployed (h=0)")
plot(prim_sep.a_vals,prim_sep.a_vals,color="black",linewidth=1.0)
xlabel("a")
ylabel("g(a,s,h)")
legend(loc="lower right")
title("Policy Functions (Separating - No Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,5))
ax[:set_xlim]((-0.525,5))

polfig1 = figure()
plot(a_vals_policy_emp1_sep,policy_emp1_sep,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_policy_unemp1_sep,policy_unemp1_sep,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
plot(a_vals_policy_emp1_sep,a_vals_policy_emp1_sep,color="black",linewidth=1.0)
xlabel("a")
ylabel("g(a,s,h)")
legend(loc="lower right")
title("Policy Functions (Separating - Bankruptcy)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,5))
ax[:set_xlim]((0,5))

# Plot default decision rule

decfig = figure()
plot(prim_sep.a_vals,results_sep.d0[:,1],color="blue",linewidth=2.0,label="Employed")
plot(prim_sep.a_vals,results_sep.d0[:,2],color="red",linewidth=2.0,label="Unemployed")
xlabel("a")
ylabel("d(a,s,0)")
legend(loc="lower right")
title("Default Decision Rule (Separating)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,2))
ax[:set_xlim]((-0.525,5))

# Bond prices

bondfig = figure()
plot(prim_sep.a_vals,prim_sep.q_menu[:,1],color="blue",linewidth=2.0,label="Employed")
plot(prim_sep.a_vals,prim_sep.q_menu[:,2],color="red",linewidth=2.0,label="Unemployed")
xlabel("a")
ylabel("q(a,s,0)")
legend(loc="lower right")
title("Bond Prices (Separating)")
ax = PyPlot.gca()
ax[:set_ylim]((0,1))
ax[:set_xlim]((-0.525,0))

# Interest Rates

intfig_emp = figure()
bar(int_rate_emp,int_dist_emp,width=0.02)
title("Distribution of Interest Rates (Separating - Employed)")
legend(loc="upper right")

intfig_unemp = figure()
PyPlot.bar(int_rate_unemp,int_dist_unemp,width=0.02)
title("Distribution of Interest Rates (Separating - Unemployed)")
legend(loc="upper right")

# Distribution

distfig1 = figure()
PyPlot.bar(prim_sep.a_vals,results_sep.statdist[1:prim_sep.a_size],
  width=0.1,alpha=0.5,color="blue",label="mu(a,s=1,h=0)")
title("Distribution - Separating (Employed/No-Bankruptcy)")
legend(loc="upper right")

distfig2 = figure()
PyPlot.bar(prim_sep.a_vals,results_sep.statdist[prim_sep.a_size+1:prim_sep.N],
  width=0.1,alpha=0.5,color="red",label="mu(a,s=0.05,h=0)")
title("Distribution - Separating (Unemployed/No-Bankruptcy)")
legend(loc="upper right")

distfig3 = figure()
PyPlot.bar(prim_sep.a_vals,results_sep.statdist[prim_sep.N+1:prim_sep.N+prim_sep.a_size],
  width=0.1,alpha=0.5,color="green",label="mu(a,s=1,h=1)")
title("Distribution - Separating (Employed/Bankruptcy)")
legend(loc="upper right")

distfig4 = figure()
PyPlot.bar(prim_sep.a_vals,results_sep.statdist[prim_sep.N+prim_sep.a_size+1:prim_sep.N*2],width=0.1,alpha=0.5,color="yellow",label="mu(a,s=0.05,h=1)")
title("Distribution - Separating (Unemployed/Bankruptcy)")
legend(loc="upper right")
