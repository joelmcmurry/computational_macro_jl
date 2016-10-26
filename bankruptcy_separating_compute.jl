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
    println("Iter: ", i, " Max Profits: ", maximum(abs(profits_sep)))

    # Adjust q (and stop if asset market clears)
    if maximum(abs(profits_sep)) < epsilon
        break
    else
      for state_today in 1:prim.s_size
        for asset_tomorrow in 1:prim.a_size
          if profits_sep[asset_tomorrow,state_today] > 0.00 # q too large
            qupper[asset_tomorrow,state_today] =
              prim.q_menu[asset_tomorrow,state_today]
          elseif profits_sep[asset_tomorrow,state_today] < 0.00 # q too small
            qlower[asset_tomorrow,state_today] =
              prim.q_menu[asset_tomorrow,state_today]
          end
        end
      end
    end

    # Update primitives given new q menu
    q_menu = (qlower + qupper)./2
    prim.q_menu = q_menu

    # Update guess for value function
    v0 = results.Tv0
    v1 = results.Tv1

  end

profits_sep, q_menu, prim, results

end

tic()
results = compute_separating(max_iter=100,a_size=500)
toc()

sep_prim = results[3]
sep_results = results[4]

#= Since policy functions are not defined over default regions
need to trim index arrays and only return non-default policies. Also need to
construct matching asset values for plotting =#

policyindex_emp1_sep = sep_results.sigma1[:,1][sep_results.sigma1[:,1].!=0]
  a_vals_policy_emp1_sep = sep_prim.a_vals[sep_results.sigma1[:,1].!=0]
policyindex_unemp1_sep = sep_results.sigma1[:,2][sep_results.sigma1[:,2].!=0]
  a_vals_policy_unemp1_sep = sep_prim.a_vals[sep_results.sigma1[:,2].!=0]

# Return values for policy functions and no-bankrupt value functions

policy_emp0_sep = sep_prim.a_vals[sep_results.sigma0[:,1]]
policy_unemp0_sep = sep_prim.a_vals[sep_results.sigma0[:,2]]
policy_emp1_sep = sep_prim.a_vals[policyindex_emp1_sep]
policy_unemp1_sep = sep_prim.a_vals[policyindex_unemp1_sep]
value_emp0_sep = sep_results.Tv0[:,1]
value_unemp0_sep = sep_results.Tv0[:,2]

#= Value functions only defined for bankrupt histories over positive assets.
Trim value function arrays and construct corresponding asset values =#

value_emp1_sep = sep_results.Tv1[:,1][sep_results.Tv1[:,1].!=-Inf]
  a_vals_value_emp1_sep = sep_prim.a_vals[sep_results.Tv1[:,1].!=-Inf]
value_unemp1_sep = sep_results.Tv1[:,2][sep_results.Tv1[:,2].!=-Inf]
  a_vals_value_unemp1_sep = sep_prim.a_vals[sep_results.Tv1[:,2].!=-Inf]

# Plot value functions

valfig = figure()
plot(sep_prim.a_vals,value_emp0_sep,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(sep_prim.a_vals,value_unemp0_sep,color="red",linewidth=2.0,label="Unemployed (h=0)")
plot(a_vals_value_emp1_sep,value_emp1_sep,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_value_unemp1_sep,value_unemp1_sep,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
xlabel("a")
ylabel("v(a,s,h)")
legend(loc="lower right")
title("Value Functions (Pooling)")
ax = PyPlot.gca()
ax[:set_ylim]((-20,2))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/valuefunctions_sep.pgf")

# Plot value function

polfig = figure()
plot(sep_prim.a_vals,policy_emp0_sep,color="blue",linewidth=2.0,label="Employed (h=0)")
plot(sep_prim.a_vals,policy_unemp0_sep,color="red",linewidth=2.0,label="Unemployed (h=0)")
plot(a_vals_policy_emp1_sep,policy_emp1_sep,color="green",linewidth=2.0,label="Employed (h=1)")
plot(a_vals_policy_unemp1_sep,policy_unemp1_sep,color="yellow",linewidth=2.0,label="Unemployed (h=1)")
plot(sep_prim.a_vals,sep_prim.a_vals,color="black",linewidth=1.0)
xlabel("a")
ylabel("g(a,s,h)")
legend(loc="lower right")
title("Policy Functions (Pooling)")
ax = PyPlot.gca()
ax[:set_ylim]((-1,5))
ax[:set_xlim]((-0.525,5))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4b/Pictures/policyfunctions_sep.pgf")
