#=
Program Name: huggett.jl
DESCRIBE THIS STUFF
=#

using Optim: optimize
using Grid: CoordInterpGrid, BCnan, InterpLinear

## Parameters
beta = 0.9932
alpha = 1.5

## Exogeneous earnings values (employed,unemployed) and Markov process
e = 1
u = 0.5
dist_e = [0.97 (1-0.97)]
dist_u = [0.5 (1-0.5)]

## Asset Grid
a_max = 5
a_min = -2
grid_size = 100
a_grid = linspace(a_min,a_max,grid_size)

## Bellman Operator

function bellman_operator(v_e,v_u,s,q::Float64,cond_dist::Array{Float64,2})
    Av_e = CoordInterpGrid(a_grid, v_e, BCnan, InterpLinear)
    Av_u = CoordInterpGrid(a_grid, v_u, BCnan, InterpLinear)

    Tv = zeros(grid_size)

    for (i, a) in enumerate(a_grid)
        objective(aprime) = -( (1/(1-alpha))*((e+a-q*aprime)-1)
          + beta*((cond_dist[1]*Av_e[aprime] +
          cond_dist[2]*Av_u[aprime])) )
        res = optimize(objective, max(-u/(1-q),-2), a_max)
        Tw[i] = - objective(res.minimum)
    end
    return Tw
end
