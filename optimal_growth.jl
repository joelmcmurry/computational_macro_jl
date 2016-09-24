#=
Program Name: optimal_growth.jl
This program generates the value function and decision
rules for a nonstochastic growth model.
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

function bellman_operator(w)
    Aw = CoordInterpGrid(grid, w, BCnan, InterpLinear)

    Tw = zeros(grid_size)

    for (i, k) in enumerate(grid)
        objective(kprime) = - log(k^alpha + (1-delta)*k-kprime) - beta * Aw[kprime]
        res = optimize(objective, 0, k^alpha + (1-delta)*k)
        Tw[i] = - objective(res.minimum)
    end
    return Tw
end

## Find a fixed point of the Bellman Operator (it is a contraction so we are guaranteed existence)

function fixed_point(T::Function; err_tol=1e-4,
                                 max_iter=1000, verbose=true, print_skip=10)
    w = zeros(grid)
    iterate = 0
    error = err_tol + 1
    while iterate < max_iter && error > err_tol
        w_next = T(w)
        iterate += 1
        error = Base.maxabs(w_next - w)
        if verbose
            if iterate % print_skip == 0
                println("Compute iterate $iterate with error $error")
            end
        end
        w = w_next
    end

    if iterate < max_iter && verbose
        println("Converged in $iterate steps")
    elseif iterate == max_iter
        warn("max_iter exceeded in fixed_point")
    end

    return w
end

v_star = fixed_point(bellman_operator)

## Obtain policy function

function policy_function(w)
    Aw = CoordInterpGrid(grid, w, BCnan, InterpLinear)

    policy = zeros(grid_size)

    for (i, k) in enumerate(grid)
        objective(kprime) = - log(k^alpha + (1-delta)*k-kprime) - beta * Aw[kprime]
        res = optimize(objective, 0, k^alpha + (1-delta)*k)
        policy[i] = res.minimum
    end
    return policy
end

policyfunction = policy_function(v_star)

plot(grid,v_star,color="blue",linewidth=2.0,label="Value Function")
plot(grid,policyfunction,color="blue",linewidth=2.0,label="Policy Function")
