#=
Program Name: optimal_growth_stochastic.jl
This program generates the value function and decision
rules for a stochastic growth model.
Adapted from Sargent and Stachurski Quantitative
Economics Lectures (original authors: Spencer Lyon,
Victoria Gregory)
=#

using Optim: optimize
using Grid: CoordInterpGrid, BCnan, InterpLinear
using PyPlot

## Parameters
beta = 0.99
delta = 0.025
alpha = 0.36

## Stochastic Production Function Parameters
z_good = 1.25
z_bad = 0.2
dist_good = [0.977 (1-0.977)]
dist_bad = [0.074 (1-0.074)]

## Asset Grid
grid_upper = 45
grid_size = 1800
grid = 1e-6:(grid_upper-1e-6)/(grid_size-1):grid_upper

#= Bellman operator

The Bellman Operator function takes as an input the interpolated value function w
defined on the grid points and solves max[u(k,k') + beta*w(k')] where k' is chosen
from the interpolated grid points
Exploit the monotonicity of the policy rule to only search for k' > k
=#

function bellman_operator(w_good,w_bad,z,cond_dist::Array{Float64,2})
    Aw_good = CoordInterpGrid(grid, w_good, BCnan, InterpLinear)
    Aw_bad = CoordInterpGrid(grid, w_bad, BCnan, InterpLinear)

    Tw = zeros(grid_size)

    for (i, k) in enumerate(grid)
        objective(kprime) = - log(z*(k^alpha) + (1-delta)*k-kprime) -
          beta * (cond_dist[1]*Aw_good[kprime] +
          cond_dist[2]*Aw_bad[kprime])
        res = optimize(objective, 0, z*(k^alpha) + (1-delta)*k)
        Tw[i] = - objective(res.minimum)
    end
    return Tw
end

## Find a fixed point of the Bellman Operator

function fixed_point(T::Function; err_tol=1e-4,
                                 max_iter=1000, verbose=true, print_skip=10)
    w_good = zeros(grid)
    w_bad = zeros(grid)
    iterate = 0
    error = err_tol + 1
    while iterate < max_iter && error > err_tol
        w_good_next = T(w_good,w_bad,z_good,dist_good)
        w_bad_next = T(w_bad,w_bad,z_bad,dist_bad)
        iterate += 1
        error_good = Base.maxabs(w_good_next - w_good)
        error_bad = Base.maxabs(w_bad_next - w_bad)
        error = max(error_good,error_bad)
        if verbose
            if iterate % print_skip == 0
                println("Compute iterate $iterate with error $error")
            end
        end
        w_good = w_good_next
        w_bad = w_bad_next
    end

    if iterate < max_iter && verbose
        println("Converged in $iterate steps")
    elseif iterate == max_iter
        warn("max_iter exceeded in fixed_point")
    end

    return w_good, w_bad
end

v_star = fixed_point(bellman_operator)

## Obtain policy function

function policy_function(w_good,w_bad,z,cond_dist::Array{Float64,2})
    Aw_good = CoordInterpGrid(grid, w_good, BCnan, InterpLinear)
    Aw_bad = CoordInterpGrid(grid, w_bad, BCnan, InterpLinear)

    policy = zeros(grid_size)

    for (i, k) in enumerate(grid)
        objective(kprime) = - log(z*(k^alpha) + (1-delta)*k-kprime) -
          beta * (cond_dist[1]*Aw_good[kprime] +
          cond_dist[2]*Aw_bad[kprime])
        res = optimize(objective, 0, z*(k^alpha) + (1-delta)*k)
        policy[i] = res.minimum
    end
    return policy
end

policyfunction_good = policy_function(v_star[1],v_star[2]
,z_good,dist_good)
policyfunction_bad = policy_function(v_star[1],v_star[2]
,z_bad,dist_bad)

## Plots

  ## Value Functions

  valfig = figure()
  plot(grid,v_star[1],color="blue",linewidth=2.0,label="Good State")
  plot(grid,v_star[2],color="red",linewidth=2.0,label="Bad State")
  xlabel("K")
  ylabel("V(K|Z)")
  legend(loc="lower right")
  title("Value Functions")
  ax = PyPlot.gca()
  ax[:set_ylim]((-175,0))
  savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Week 2/Pictures/valuefunctions.pgf")

  ## Policy Functions

  polfig = figure()
  plot(grid,policyfunction_good,color="blue",linewidth=2.0,label="Good State")
  plot(grid,policyfunction_bad,color="red",linewidth=2.0,label="Bad State")
  xlabel("K")
  ylabel("K'(K)")
  legend(loc="lower right")
  title("Policy Functions")
  ax = PyPlot.gca()
  ax[:set_ylim]((0,45))
  savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Week 2/Pictures/policyfunctions.pgf")

  ## K' - K

  differencefig = figure()
  plot(grid,policyfunction_good-grid,color="blue",linewidth=2.0,label="Good State")
  plot(grid,policyfunction_bad-grid,color="red",linewidth=2.0,label="Bad State")
  xlabel("K")
  ylabel("K'-K")
  legend(loc="lower right")
  title("Change in Policy Functions (K'-K)")
  ax = PyPlot.gca()
  ax[:set_ylim]((-2.5,2.5))
  savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/Week 2/Pictures/deltapolicyfunctions.pgf")
