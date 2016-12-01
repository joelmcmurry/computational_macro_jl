#=
Program Name: krusell_smith_compute.jl
Runs Krusell-Smith model
=#

using PyPlot

include("krusell_smith_model.jl")

# initialize model primitives

prim = Primitives()

## Start in the good state and simulate sequence of T aggregate shocks

agg_shock_index = zeros(Int64,prim.T)
agg_shock_vals = zeros(Float64,prim.T)

agg_shock_index[1] = 1
agg_shock_vals[1] = prim.z[1]
for t in 2:prim.T
  if rand() <= prim.transmatagg[agg_shock_index[t-1],1]
    agg_shock_index[t] = 1
    agg_shock_vals[t] = prim.z[1]
  else
    agg_shock_index[t] = 2
    agg_shock_vals[t] = prim.z[2]
  end
end
