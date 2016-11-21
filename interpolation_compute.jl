#=
Program Name: interpolation_compute.jl
Computes optimal growth through interpolation and generates graphs
=#

using PyPlot
using QuantEcon: meshgrid
using Optim: optimize
using Interpolations

include("interpolation_model.jl")

## Find value function and policy function

const prim = Primitives()

# Linear Interpolation

tic()
res_linear = SolveProgram(prim,"Linear")
toc()

# Cubic in k-dimension and linear in k-dimension

tic()
res_cubiclinear = SolveProgram(prim,"CubicLinear")
toc()

## Plots

# Contour plots

linvalfig = figure()
ax = linvalfig[:gca](projection="3d")
ax[:set_zlim](-105,-80)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Value (linear)")
ax[:plot_surface](xgrid,ygrid,res_linear.v',rstride=2,
  cstride=2,cmap=ColorMap("jet"),alpha=0.7,linewidth=0.25)
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/val_linear.pgf")

linpolfig = figure()
ax = linpolfig[:gca](projection="3d")
ax[:set_zlim](0,0.5)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Policy (linear)")
ax[:plot_surface](xgrid,ygrid,res_linear.sigma',rstride=2,
  cstride=2,cmap=ColorMap("jet"),alpha=0.7,linewidth=0.25)
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/pol_linear.pgf")

cubvalfig = figure()
ax = cubvalfig[:gca](projection="3d")
ax[:set_zlim](-105,-80)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Value (cubic, linear)")
ax[:plot_surface](xgrid,ygrid,res_cubiclinear.v',rstride=2,
  cstride=2,cmap=ColorMap("jet"),alpha=0.7,linewidth=0.25)
  savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/val_cubiclinear.pgf")

cubpolfig = figure()
ax = cubpolfig[:gca](projection="3d")
ax[:set_zlim](0,0.5)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Policy (cubic, linear)")
ax[:plot_surface](xgrid,ygrid,res_cubiclinear.sigma',rstride=2,
  cstride=2,cmap=ColorMap("jet"),alpha=0.7,linewidth=0.25)
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/pol_cubiclinear.pgf")

# Representative K levels for LaTeX file

repvalfig_linear = figure()
plot(prim.k_vals[1:prim.k_size],res_linear.v[:,1],color="blue",label="K=0.00",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_linear.v[:,5],color="red",label="K=0.154",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_linear.v[:,10],color="green",label="K=0.159",linewidth=2.0)
xlabel("k")
ylabel("v(k;K=k)")
legend(loc="lower right")
title("Value (linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/repval_linear.pgf")

reppolfig_linear = figure()
plot(prim.k_vals[1:prim.k_size],res_linear.sigma[:,1],color="blue",label="K=0.00",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_linear.sigma[:,5],color="red",label="K=0.154",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_linear.sigma[:,10],color="green",label="K=0.159",linewidth=2.0)
xlabel("k")
ylabel("k'(k;K=k)")
legend(loc="lower right")
title("Policy (linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/reppol_linear.pgf")

repvalfig_cubiclinear = figure()
plot(prim.k_vals[1:prim.k_size],res_cubiclinear.v[:,1],color="blue",label="K=0.00",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_cubiclinear.v[:,5],color="red",label="K=0.154",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_cubiclinear.v[:,10],color="green",label="K=0.159",linewidth=2.0)
xlabel("k")
ylabel("v(k;K=k)")
legend(loc="lower right")
title("Value (cubic, linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/repval_cubiclinear.pgf")

reppolfig_cubiclinear = figure()
plot(prim.k_vals[1:prim.k_size],res_cubiclinear.sigma[:,1],color="blue",label="K=0.00",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_cubiclinear.sigma[:,5],color="red",label="K=0.154",linewidth=2.0)
plot(prim.k_vals[1:prim.k_size],res_cubiclinear.sigma[:,10],color="green",label="K=0.159",linewidth=2.0)
xlabel("k")
ylabel("k'(k;K=k)")
legend(loc="lower right")
title("Policy (cubic, linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/reppol_cubiclinear.pgf")


## Isolate calculated value functions and policy functions for k=K

itp_k = interpolate(prim.k_vals,BSpline(Linear()),OnGrid())
itp_value_linear = interpolate(res_linear.v,BSpline(Cubic(Line())),OnGrid())
itp_value_cubiclinear = interpolate(res_cubiclinear.v,(BSpline(Cubic(Line())),BSpline(Linear())),OnGrid())
itp_policy_linear = interpolate(res_linear.sigma,BSpline(Cubic(Line())),OnGrid())
itp_policy_cubiclinear = interpolate(res_cubiclinear.sigma,(BSpline(Cubic(Line())),BSpline(Linear())),OnGrid())

value_linear = fill(-Inf,prim.K_size)
value_cubiclinear = fill(-Inf,prim.K_size)
policy_linear = zeros(Float64,prim.K_size)
policy_cubiclinear = zeros(Float64,prim.K_size)

for K_index in 1:prim.K_size
    findmatch(k_index)=abs(itp_k[k_index]-prim.K_vals[K_index])
    k_index_match = optimize(k_index->findmatch(k_index),1.0,100.0).minimum
    value_linear[K_index] = itp_value_linear[k_index_match,K_index]
    value_cubiclinear[K_index] = itp_value_cubiclinear[k_index_match,K_index]
    policy_linear[K_index] = itp_policy_linear[k_index_match,K_index]
    policy_cubiclinear[K_index] = itp_policy_cubiclinear[k_index_match,K_index]
end

# Plot k=K for interpolated

keqKval_linear = figure()
plot(prim.K_vals[1:prim.K_size],value_linear,color="red",linewidth=2.0)
xlabel("K")
ylabel("v(k;K=k)")
legend(loc="lower right")
title("Value k=K (linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/keqK_val_linear.pgf")

keqKpol_linear = figure()
plot(prim.K_vals[1:prim.K_size],policy_linear,color="red",linewidth=2.0)
xlabel("K")
ylabel("k'(k;K=k)")
legend(loc="lower right")
title("Policy k=K (linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/keqK_policy_linear.pgf")

keqKval_cubiclinear = figure()
plot(prim.K_vals[1:prim.K_size],value_cubiclinear,color="green",linewidth=2.0)
xlabel("K")
ylabel("v(k;K=k)")
legend(loc="lower right")
title("Value k=K (cubic, linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/keqK_val_cubiclinear.pgf")

keqKpol_cubiclinear = figure()
plot(prim.K_vals[1:prim.K_size],policy_cubiclinear,color="green",linewidth=2.0)
xlabel("K")
ylabel("k'(k;K=k)")
legend(loc="lower right")
title("Policy k=K (cubic, linear)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/keqK_policy_cubiclinear.pgf")

## Calculate closed form solutions for k=K

A = (1/(1-prim.beta))*
  ((prim.beta*prim.alpha/(1-prim.beta*prim.alpha))*
    log(prim.beta*prim.alpha)+log(1-prim.beta*prim.alpha))
B = prim.alpha/(1-prim.alpha*prim.beta)
value_closed = A+B*log(prim.k_vals)
policy_closed = (prim.k_vals.^prim.alpha)*prim.beta*prim.alpha

closedvalfig = figure()
plot(prim.k_vals[2:prim.k_size],value_closed[2:prim.k_size],color="blue",linewidth=2.0)
xlabel("k")
ylabel("v(k;K=k)")
legend(loc="lower right")
title("Value (closed form)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/closed_value.pgf")

closedpolfig = figure()
plot(prim.k_vals[2:prim.k_size],policy_closed[2:prim.k_size],color="blue",linewidth=2.0)
xlabel("k")
ylabel("k'(k;K=k)")
legend(loc="lower right")
title("Policy (closed form)")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS7/Pictures/closed_policy.pgf")
