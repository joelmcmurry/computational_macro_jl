#=
Program Name: conesa_krueger_compute.jl
Runs Conesa-Krueger model
=#

using PyPlot

include("conesa_krueger_model.jl")

## Initialize Primitives

prim = Primitives(a_size=1000,a_max=100.00)

#= Solve worker problem and return value functions and policy functions
for 50-year old and a 20-year old =#
tic()
v_20, v_50, policy_20, policy_50, labor_supply_20 = SolveProgram(prim)
toc()

#= Graphs =#

v50fig = figure()
plot(prim.a_vals,v_50,color="blue",linewidth=2.0)
xlabel("a")
ylabel("value")
legend(loc="lower right")
title("Value Function (Age 50)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS5/Pictures/v50.pgf")

policy20fig = figure()
plot(prim.a_vals,policy_20[:,1],color="blue",linewidth=2.0,label="High Productivity")
plot(prim.a_vals,policy_20[:,2],color="red",linewidth=2.0,label="Low Productivity")
xlabel("a")
ylabel("a'(a,z)")
legend(loc="lower right")
title("Policy Function (Age 20)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS5/Pictures/policy20.pgf")
