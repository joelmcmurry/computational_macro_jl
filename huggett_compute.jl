#=
Program Name: huggett_compute.jl
=#

using PyPlot

include("huggett_new.jl")

function compute_huggett(;q0=0.9,max_iter=100,
  max_iter_vfi=2000,epsilon=1e-2,a_size=500)

  #= Instantiate Huggett model with a starting guess for
  discount bond price and an initialized net_asset level.
  We will want to keep track of the price last period 'pmin1'
  for the convergence process=#

  q = q0
  qmin1 = 1
  net_assets = 10.0

  huggett = Huggett(q=q,a_size=a_size)

  ## Create Dynamic Program

  huggettdp = DiscreteProgram(huggett.R,huggett.Q,huggett.beta)

  ## Solve Dynamic Program given initial q

  huggettres = SolveProgram(huggettdp,max_iter_vfi=max_iter_vfi)

  ## Calculate net asset holdings for initial q

  net_assets = dot(huggett.a_vals[huggettres.sigma],
    huggettres.statdist)

  # Print initial net assets and discount bond price
  println("Iter: ", 0, " Net Assets: ", net_assets," q: ", q)

  for i in 1:max_iter

    # Adjust q (and stop if asset market clears)
    if abs(net_assets) < epsilon
        break
    elseif net_assets > 0 # q too small
      if q > qmin1
        qprime = q + abs(q - qmin1)/2
      else
        qprime = (q + qmin1)/2
      end
    else # q too big
      if q > qmin1
        qprime = (q + qmin1)/2
      else
        qprime = q - abs(q - qmin1)/2
      end
    end
    qmin1 = q
    q = qprime

    # Re-initialize current return matrix
    huggett.R = fill(-Inf,huggett.N,a_size)

    # Update current return matrix given new q
    curr_return!(huggett,q)

    # Replace current return matrix in dynamic program
    huggettdp.R = huggett.R

    # Solve dynamic program given new q
    huggettres = SolveProgram(huggettdp,max_iter_vfi=max_iter_vfi)

    # Calculate net asset holdings using stationary distribution
    net_assets = dot(huggett.a_vals[huggettres.sigma],
      huggettres.statdist)

    # Print iteration, net assets, and discount bond price
    println("Iter: ", i, " Net Assets: ", net_assets," q: ", q)

  end

out_value = huggettres.Tv
out_policy = huggettres.sigma
out_statdist = huggettres.statdist
out_assethold = huggett.a_vals[huggettres.sigma]

return net_assets, q, out_value, out_policy, out_statdist, huggett, huggettres

end

tic()
results = compute_huggett(max_iter=100,a_size=100,q0=0.9)
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
