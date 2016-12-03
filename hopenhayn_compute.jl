#=
Program Name: hopenhayn_compute.jl
Runs Hopenhayn model
=#

using PyPlot

include("hopenhayn_model.jl")

# initialize model primitives

const prim = Primitives()

function hopenhayn_compute(prim::Primitives;max_iter=100,ec_tol=1e-2)

  # initialize results objects

  res = Results(prim)

  ## Find price that satisfied entry condition

  p_lower = 0.0 # lower guess for p
  p_upper = 100.0 # upper guess for p

  for i in 1:max_iter

    # find decision rules and value functions

    res = DecisionRules(prim)

    # calculate entry condition EC(p)

    EC = dot(prim.nu,res.TWe)/prim.p - prim.ce

    # update price and stop when entry condition is close to satisfied

    println("Iter: ", i, " Price: ", prim.p," EC: ", EC)

    if abs(EC) < ec_tol
      break
    end

    if EC > 0
      p_upper = prim.p # lower price
    else
      p_lower = prim.p# raise price
    end
    p_new = (p_upper + p_lower)/2

    prim.p = p_new

  end

  ## Find stationary distribution and entrant measure

  # guess M = 1

  res.M = 1

  # find stationary distribution given guess for M

end
