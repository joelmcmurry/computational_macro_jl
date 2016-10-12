#=
Program Name: huggett_compute.jl
=#

using PyPlot

include("huggett_new.jl")

function compute_huggett(;q0=0.9,max_iter=100,
  max_iter_vfi=2000,epsilon=1e-2,a_size=500)

  #= Instantiate primitives of model with a starting range for
  discount bond price=#

  qlower = q0
  qupper = 1

  q = (qlower + qupper)/2

  # Initialize primitives
  prim = Primitives(q=q,a_size=a_size)

  # Initial guess for value function
  v = zeros(prim.a_size,prim.s_size)

  # Initialize net assets
  net_assets::Float64 = 10.0

  # Initialize results structure
  results = Results(prim)

  for i in 1:max_iter

    # Solve dynamic program given new q
    results = SolveProgram(prim,v,max_iter_vfi=max_iter_vfi)

    # Calculate net assets given new q
    net_assets = 0

    for state in 1:prim.N
      holdings = results.statdist[state]*
      prim.a_vals[results.sigma[prim.a_s_indices[state,1],
      prim.a_s_indices[state,2]]] #asset choice times density at that state
      net_assets += holdings
    end

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Net Assets: ", net_assets," q: ", q)

    # Adjust q (and stop if asset market clears)
    if abs(net_assets) < epsilon
        break
    elseif net_assets > 0 # q too small
      qlower = q
    else # q too big
      qupper = q
    end

    q = (qlower + qupper)/2

    # Update primitives given new q
    prim.q = q

    # Update guess for value function
    v = results.Tv

  end

net_assets, q, prim, results

end

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
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Week 3/Pictures/policyfunctions.pgf")

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
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Week 3/Pictures/valuefunctions.pgf")


# Plot stationary distribution

distfig = figure()
bar(huggett.a_vals,huggett_results.statdist[1:huggett.a_size],
  color="blue",label="Employed")
bar(huggett.a_vals,huggett_results.statdist[huggett.a_size+1:huggett.N],
  color="red",label="Unemployed")
title("Wealth Distribution")
legend(loc="upper right")
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Week 3/Pictures/stationarydistributions.pgf")

# Plot Lorenz Curve

wealth_vals = huggett.a_s_vals[:,1]+huggett.a_s_vals[:,2]
wealth_held = wealth_vals .* huggett_results.statdist
test = collect(wealth_held, wealth_vals)

sort then use cumsum and multiply by total to get percentages

## Calculate Consumption Equivalent

# Plot Consumption Equivalent
