#=
Program Name: interpolation_compute.jl
Computes optimal growth through interpolation and generates graphs
=#

using PyPlot
using QuantEcon: meshgrid

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

# bullshit
x = prim.k_vals
y = prim.K_vals
z = res_linear.v

testfig = figure()
plot(prim.k_vals,z[:,50],color="blue",linewidth=2.0,label="test")
ax = PyPlot.gca()

testfig1 = figure()
surface(x,y,z,alpha=0.7)

testx = linspace(1,100,100)
testy = linspace(1,100,100)

#bullshit end

# moderate bullshit

fig = figure()
ax = fig.gca(projection='3d')
ax.plot_surface(prim.k_vals,prim.K_vals,res_linear.v,rstride=8,cstride=8,alpha=0.3)
ax.set_xlabel('k')
ax.set_xlim(0,0.5)
ax.set_xlabel('K')
ax.set_xlim(0.15,0.25)
ax.set_xlabel('Value')
ax.set_xlim(-105,-80)
show()

# moderate bullshit end

linvalfig = figure()
ax = linvalfig[:gca](projection="3d")
ax[:set_zlim](-105,-80)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Value (linear)")
ax[:plot_surface](ygrid,xgrid,res_linear.v,rstride=2,
  cstride=2,cmap=ColorMap("jet"),alpha=0.7,linewidth=0.25)

linpolfig = figure()
ax = linpolfig[:gca](projection="3d")
ax[:set_zlim](0,0.5)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Policy (linear)")
ax[:plot_surface](xgrid,ygrid,res_linear.sigma)

cubvalfig = figure()
ax = cubvalfig[:gca](projection="3d")
ax[:set_zlim](-105,-80)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Value (cubic, linear)")
ax[:plot_surface](xgrid,ygrid,res_cubiclinear.v)

cubpolfig = figure()
ax = cubpolfig[:gca](projection="3d")
ax[:set_zlim](0,0.5)
xgrid, ygrid = meshgrid(prim.k_vals,prim.K_vals)
title("Policy (cubic, linear)")
ax[:plot_surface](xgrid,ygrid,res_cubiclinear.sigma)

## Isolate calculated value functions and policy functions for k=K

value_linear =

## Calculate closed form solutions for k=K

A = (1/(1-prim.beta))*
  ((prim.beta*prim.alpha/(1-prim.beta*prim.alpha))*
    log(prim.beta*prim.alpha)+log(1-prim.beta*prim.alpha))
B = prim.alpha/(1-prim.alpha*prim.beta)
value_closed = A+B*log(prim.k_vals)
policy_closed = (prim.k_vals.^prim.alpha)*prim.beta*prim.alpha


teststuff=0.0
for j in 1:100
  for i in 2:100
    if res_linear.v[i,j]<res_linear.v[i-1,j]
      teststuff+=1.0
    end
  end
end
