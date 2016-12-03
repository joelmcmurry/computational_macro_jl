#=
Program Name: hopenhayn_compute.jl
Runs Hopenhayn model
=#

using PyPlot

include("hopenhayn_model.jl")

# initialize model primitives

const prim = Primitives()

function hopenhayn_compute(prim::Primitives;max_iter=100,epsilon=1e-2)

  # initialize results objects

  res = Results(prim)

  # find price that satisfied entry condition

  for i in 1:max_iter
    
  end

end
