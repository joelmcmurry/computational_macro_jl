#=
Program Name: huggett_compute.jl
=#

using PyPlot

include("huggett_new.jl")

function compute_huggett(;q0=0.9,max_iter=100,
  max_iter_vfi=2000,epsilon=1e-2,a_size=500)

  #= Instantiate primitives of model with a starting guess for
  discount bond price. We will want to keep track of the price
  last period 'pmin1' for the convergence process=#

  q = q0
  qmin1 = 1

  prim = Primitives(q=q,a_size=a_size)

  ## Solve dynamic program given initial q

  results = SolveProgram(prim,max_iter_vfi=max_iter_vfi)

  ## Calculate net asset holdings for initial q

  net_assets = 0

  for state in 1:N
    holdings = results.statdist[state]*
    prim.a_vals[results.sigma[prim.a_s_indices[state,1],
    prim.a_s_indices[state,2]]] #asset choice times density at that state
    net_assets += holdings
  end

  # Print initial net assets and discount bond price
  println("Iter: ", 0, " Net Assets: ", net_assets," q: ", q)

  for i in 1:max_iter

    # Adjust q (and stop if asset market clears)
    if abs(net_assets) < epsilon
        break
    elseif net_assets > 0 # q too small
      if q > qmin1
        qprime = q + abs(q - qmin1)
      else
        qprime = (q + qmin1)/2
      end
    else # q too big
      if q > qmin1
        qprime = (q + qmin1)/2
      else
        qprime = q - abs(q - qmin1)
      end
    end
    qmin1 = q
    q = qprime

    # Update primitives given new q
    prim.q = q

    # Solve dynamic program given new q
    results = SolveProgram(prim,max_iter_vfi=max_iter_vfi)

    # Calculate net assets given new q

    net_assets = 0

    for state in 1:N
      holdings = results.statdist[state]*
      prim.a_vals[results.sigma[prim.a_s_indices[state,1],
      prim.a_s_indices[state,2]]] #asset choice times density at that state
      net_assets += holdings
    end

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Net Assets: ", net_assets," q: ", q)

  end

return net_assets, q, results

end

tic()
huggett_results = compute_huggett(q0=0.8)
toc()

huggett = results[6]
huggettres = results[7]

policy_emp = huggett.a_vals[huggettres.sigma[1:huggett.a_size]]
policy_unemp = huggett.a_vals[huggettres.sigma[huggett.a_size+1:huggett.N]]

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

# Plot Lorenz Curve
