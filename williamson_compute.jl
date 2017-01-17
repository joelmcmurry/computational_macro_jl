#=
Program Name: williamson_compute.jl
Computes Williamson (1998) model
=#

using PyPlot

include("williamson_model.jl")

# initialize model primitives

prim = Primitives(w_size=50,w_size_itp=500)

function williamson_compute(prim::Primitives;
    max_iter_ec=200,ec_tol=1e-2)

  # initialize results object for output

  res = Results(prim)

  ## Find discount bond price that satisfies entry condition

  q_lower = 0.0 # lower bound for q
  q_upper = 1.0 # upper bound for q

  for i in 1:max_iter_ec

    # use last value function as initial guess
    v = res.v

    # find decision rules and value functions
    res = SolveProgram(prim,v)

    # calculate entry condition EC(q)

    EC = 0.0

    for w_index in 1:prim.w_size_itp
      transfers = prim.pi*res.tau_vals_itp[w_index,1] + (1-prim.pi)*res.tau_vals_itp[w_index,2]
      EC += transfers*res.statdist[w_index]
    end

    # update price and stop when entry condition is close to satisfied

    println("Iter: ", i, " Error: ", EC-prim.ce, " Price: ", prim.q, " Beta: ", prim.beta)

    if abs(EC-prim.ce) < ec_tol
      break
    end

    if EC > prim.ce
      q_lower = prim.q # raise discount bond price
    else
      q_upper = prim.q # lower discount bond price
    end
    q_new = (q_upper + q_lower)/2

    prim.q = q_new

  end

  # Calculate agent consumption

  consumption!(prim,res)

  prim, res
end

tic()
prim, res = williamson_compute(prim)
toc()

#= Plots =#

## Principal Value

valfig = figure()
plot(prim.w_vals,res.v,color="blue",linewidth=2.0)
xlabel("Agent Utility w")
ylabel("Principal Value z")
title("Principal Value")
ax = PyPlot.gca()
ax[:set_xlim]((prim.w_min,prim.w_max))
ax[:set_ylim]((0,1.8))
#savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Final Project/Pictures/princ_value.pgf")

## Transfers

transferfig = figure()
plot(prim.w_vals,res.tau_vals[:,1],color="blue",linewidth=1.0,label="t_H")
plot(prim.w_vals,res.tau_vals[:,2],color="red",linewidth=1.0,label="t_L")
plot(prim.w_vals,zeros(prim.w_vals),color="green",linewidth=0.5,label="zero")
xlabel("Agent Utility w")
ylabel("Transfer")
title("Transfers")
legend(loc="lower right")
ax = PyPlot.gca()
ax[:set_xlim]((prim.w_min,prim.w_max))
ax[:set_ylim]((-1.5,1))
#savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Final Project/Pictures/transfers.pgf")

## Utility Promises

utilityfig = figure()
plot(prim.w_vals,res.wprime_vals[:,1],color="blue",linewidth=1.0,label="w'_H")
plot(prim.w_vals,res.wprime_vals[:,2],color="red",linewidth=1.0,label="w'_L")
plot(prim.w_vals,prim.w_vals,color="green",linewidth=0.5,label="w=w'")
xlabel("Agent Utility w")
ylabel("Agent Future Utility w'")
title("Utility Dynamics")
legend(loc="lower right")
ax = PyPlot.gca()
ax[:set_xlim]((prim.w_min,prim.w_max))
ax[:set_ylim]((prim.w_min,prim.w_max))
#savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Final Project/Pictures/utilities.pgf")

## Consumption

consfig = figure()
plot(prim.w_vals,res.wprime_vals[:,1],color="blue",linewidth=1.0,label="c_H=y_H+t_H")
plot(prim.w_vals,res.wprime_vals[:,2],color="red",linewidth=1.0,label="c_L=y_L+t_L")
xlabel("Agent Utility w")
ylabel("Agent Consumption")
title("Agent Consumption")
legend(loc="lower right")
ax = PyPlot.gca()
ax[:set_xlim]((prim.w_min,prim.w_max))
ax[:set_ylim]((0,3))
#savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Final Project/Pictures/consumption.pgf")

## Stationary Distribution

distfig = figure()
plot(prim.w_vals_itp,res.statdist,color="blue",linewidth=1.0)
xlabel("Agent Utility w")
ylabel("Density")
title("Stationary Distribution")
ax = PyPlot.gca()
ax[:set_xlim]((prim.w_min,prim.w_max))
ax[:set_ylim]((0,0.1))
#savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Final Project/Pictures/statdist.pgf")
