#=
Program Name: bankruptcy_compute.jl
Runs bankruptcy model
=#

using PyPlot

include("bankruptcy_model.jl")

function compute_pooling(;q0=0.9,max_iter=100,
  max_iter_vfi=2000,epsilon=1e-2,a_size=500)

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
  profits_pool::Float64 = 10.0

  # Initialize results structure
  results = Results(prim,v0,v1)

  for i in 1:max_iter

    # Solve dynamic program given new q_pool
    results = SolveProgram(prim,v0,v1,max_iter_vfi=max_iter_vfi)

    ## Calculate loss rate for lenders
      # Total assets length
      L =

      # Total defaulted assets
      D =

      # Loss rate
      Deltaprime = D/L

      profits_pool = q_pool - (1 - Deltaprime)/(1 + prim.r)

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Profits: ", profits_pool," q_pool: ", q_pool)

    # Adjust q (and stop if asset market clears)
    if abs(profits_pool) < epsilon
        break
    elseif profits_pool > 0 # q too smallFIGURE THIS OUT
      qlower = q_pool
    else # q too big FIGURE THIS OUT
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
#######################################
tic()
results = compute_huggett(q0=0.9932,max_iter=100,a_size=2000)
toc()

huggett = results[3]
huggett_results = results[4]

policy_emp = huggett.a_vals[huggett_results.sigma[:,1]]
policy_unemp = huggett.a_vals[huggett_results.sigma[:,2]]
value_emp = huggett_results.Tv[:,1]
value_unemp = huggett_results.Tv[:,2]

# Plot policy function

polfig = figure()
plot(huggett.a_vals,policy_emp,color="blue",linewidth=2.0,label="Employed")
plot(huggett.a_vals,policy_unemp,color="red",linewidth=2.0,label="Unemployed")
plot(huggett.a_vals,huggett.a_vals,color="green",linewidth=1.0,label="45degree")
xlabel("a")
ylabel("g(a,s)")
legend(loc="lower right")
title("Policy Functions")
ax = PyPlot.gca()
ax[:set_ylim]((-2,5))
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

## Plot Lorenz Curve

# Define wealth as assets plus earnings
wealth_vals = huggett.a_s_vals[:,1]+huggett.a_s_vals[:,2]
wealth_held = wealth_vals .* huggett_results.statdist

# Create matrix with wealth holdings density matched to the wealth value
wealth_mat = zeros(huggett.N,2)
for i in 1:huggett.N
  wealth_mat[i,1] = wealth_held[i]
  wealth_mat[i,2] = wealth_vals[i]
end

# Sort from poorest to richest
wealth_mat_sort = sortrows(wealth_mat,by=x->x[2])

# Calculate cumulative wealth holdings as percentage of total
perc_wealth = cumsum(wealth_mat_sort,1)*1/sum(wealth_held)

# Create matrix with agent density matched to the wealth value
dist_mat = zeros(huggett.N,2)
for i in 1:huggett.N
  dist_mat[i,1] = huggett_results.statdist[i]
  dist_mat[i,2] = wealth_vals[i]
end

# Sort from poorest to richest
dist_mat_sort = sortrows(dist_mat,by=x->x[2])

# Calculate cumulative fraction of agents
perc_agents = cumsum(dist_mat_sort,1)

lorenzfig = figure()
plot(perc_agents[:,1],perc_wealth[:,1],color="blue",linewidth=2.0)
plot(perc_agents[:,1],perc_agents[:,1],color="green",linewidth=1.0)
xlabel("Fraction of Agents")
ylabel("Fraction of Wealth")
title("Lorenz Curve")
ax = PyPlot.gca()
ax[:set_ylim]((-0.06,1))
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4/Pictures/lorenzcurve.pgf")

# Calculate Gini index

# Approximate integral under Lorenz curve with Riemann sums
gini_rect = zeros(huggett.N)
for i in 1:huggett.N-1
  gini_rect[i] = (perc_agents[i,1]-perc_wealth[i,1])*
    (perc_agents[i+1,1]-perc_agents[i,1])
end

gini_index = (0.5-sum(gini_rect))/0.5

## Calculate Consumption Equivalent

# Find stationary distribution of Markov transition matrix

function findstatmeasure(markov::Array{Float64,2},epsilon::Real)
  p0 = [0.5 1-0.5]
  p1 = p0
  diff = 1
  while diff > epsilon
    p1 = p0*markov
    diff = maxabs(p1 .- p0)
    p0=p1
  end
  return p1
end

statmeasure = findstatmeasure(huggett.markov,1e-6)

# Calculate welfare with complete markets
consfb = statmeasure[1]*1+statmeasure[2]*0.5
Wfb = (1/(1-huggett.beta))*(1/(1-huggett.alpha))*(1/(consfb^(huggett.alpha-1))-1)

# Calculate welfare with incomplete markets
Winc = dot(huggett_results.statdist[1:huggett.a_size],huggett_results.Tv[:,1])+
  dot(huggett_results.statdist[huggett.a_size+1:huggett.N],huggett_results.Tv[:,2])

# Calculate consumption equivalent
lambda = ((Wfb+(1/((1-huggett.alpha)*(1-huggett.beta))))^(1/(1-huggett.alpha)))*
  ((huggett_results.Tv .+ (1/((1-huggett.alpha)*(1-huggett.beta)))).^(1/(huggett.alpha-1))).-1

lambda_emp = lambda[:,1]
lambda_unemp = lambda[:,2]

# Calculate welfare gain
WG = dot(huggett_results.statdist[1:huggett.a_size],lambda_emp)+
  dot(huggett_results.statdist[huggett.a_size+1:huggett.N],lambda_unemp)

# Calculate fraction of population in favor of switching to complete markets

frac_switch = dot(lambda_emp.>=0,huggett_results.statdist[1:huggett.a_size])+
  dot(lambda_unemp.>=0,huggett_results.statdist[huggett.a_size+1:huggett.N])

# Plot Consumption Equivalent

conseqfig = figure()
plot(huggett.a_vals,lambda[:,1],color="blue",linewidth=2.0,label="Employed")
plot(huggett.a_vals,lambda[:,2],color="red",linewidth=2.0,label="Unemployed")
xlabel("a")
ylabel("lambda(a,s)")
legend(loc="lower right")
title("Consumption Equivalents")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS4/Pictures/consequiv.pgf")
