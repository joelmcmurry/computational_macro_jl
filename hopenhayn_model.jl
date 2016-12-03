#=
Program Name: hopenhayn_model.jl
Creates Hopenhayn Model and Utilities
=#

using QuantEcon: gridmake

## Create compostive type to hold model primitives

type Primitives
  beta :: Float64 ## discount rate
  theta :: Float64 ## production function curvature
  A :: Float64 ## employment disutility parameter
  cf :: Float64 ## production fixed cost period
  ce :: Float64 ## entry cost
  p :: Float64 ## price of output
  s_vals :: Array{Float64} ## productivity shock values
  s_size :: Int64 ## number of productivity values
  F :: Array{Float64} ## Markov transition for productivity shocks
  nu :: Array{Float64} ## entry distribution
  n_min :: Float64 ## minimum labor demand value
  n_max :: Float64 ## maximum labor demand value
  n_size :: Int64 ## size of labor demand grid
  n_vals :: Array{Float64} ## asset grid
  n_indices :: Array{Int64} ## indices of choices
end

#= Outer Constructor for Primitives. Supplies default values, field
names, and creates grid objects=#

function Primitives(;beta::Float64=0.8,theta::Float64=0.64,A::Float64=(1/200),
    cf::Float64=10.0,ce::Float64=15.0,p::Float64=1.0,
    n_min::Float64=1e-1,n_max::Float64=100.0,n_size::Int64=100)

  # Calibrated arrays

  s_vals = [3.98e-4 3.58 6.82 12.18 18.79]
  s_size = length(s_vals)
  nu = [0.37 0.4631 0.1102 0.0504 0.0063]

  F = [0.6598 0.2600 0.0416 0.0331 0.0055;
      0.1997 0.7201 0.0420 0.0326 0.0056;
      0.2000 0.2000 0.5555 0.0344 0.0101;
      0.2000 0.2000 0.2502 0.3397 0.0101;
      0.2000 0.2000 0.2500 0.3400 0.0100]

  # Grids

  n_vals = linspace(n_min, n_max, n_size)
  n_indices = gridmake(1:n_size)

  primitives = Primitives(beta, theta, A, cf, ce, p, s_vals, s_size,
    F, nu, n_min, n_max, n_size, n_vals, n_indices)

  return primitives

end

## Type Results which holds results of the problem

type Results
    W::Array{Float64} ## incumbent value
    TW::Array{Float64} ## incumbent image of bellman operator
    TWe::Array{Float64} ## incumbent image of bellman operator
    num_iter::Int ## VFI convergence iterations
    mu::Array{Float64} ## stationary distribution over firm size
    M::Float64 ## entrant mass
    nd::Array{Int} ## labor demand incumbent
    nde::Array{Int} ## labor demand entrant
    x::Array{Int} ## exit choice incumbent
    xe::Array{Int} ## exit choice entrant

    function Results(prim::Primitives)
        W = zeros(prim.s_size) # initialize value with zeroes
        mu = ones(prim.s_size)*(1/prim.s_size) # initialize stationary distribution with uniform
        res = new(W, similar(W), similar(W), 0, mu, 0.5,
        similar(W,Int), similar(W,Int), similar(W,Int), similar(W,Int))

        res
    end

    # Version w/ initial W, TWe
    function Results(prim::Primitives,W::Array{Float64})
        mu = ones(prim.s_size)*(1/prim.s_size) # initialize stationary distribution with uniform
        res = new(W, similar(W), similar(W), 0, mu, 0.5,
        similar(W,Int), similar(W,Int), similar(W,Int), similar(W,Int))

        res
    end
end

## Generate decision rules

# Without initial value function (will be initialized at zeros)
function DecisionRules(prim::Primitives;
  max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3)
    res = Results(prim)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi)
    res
end

# With Initial value function
function DecisionRules(prim::Primitives, W::Array{Float64};
  max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3)
    res = Results(prim, W)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi)
    res
end

#= Internal Utilities =#

## Bellman Operators

# incumbent firm

function bellman_operator_incumbent!(prim::Primitives, W::Array{Float64})
  # initialize
  TW = fill(-Inf,prim.s_size)
  nd = zeros(Int64,prim.s_size)
  x = zeros(Int64,prim.s_size)

  # find max value for each productivity level
  for s_index in 1:prim.s_size
  s = prim.s_vals[s_index]

  max_value = -Inf # initialize value

    for labor_choice in 1:prim.n_size
      n = prim.n_vals[labor_choice]

      # calculate exit value
      exit_value = prim.p*s*n^prim.theta - n - prim.p*prim.cf

      # calculate value staying in market
      value = exit_value + prim.beta*dot(prim.F[s_index,:],W)

      if value >= exit_value
        if value > max_value
          max_value = value
          nd[s_index] = labor_choice
          x[s_index] = 0
        end
      else
        if exit_value > max_value
          max_value = exit_value
          nd[s_index] = labor_choice
          x[s_index] = 1
        end
      end
    end
    TW[s_index] = max_value
  end
  TW, nd, x
end

# entrant firm

function bellman_operator_entrant!(prim::Primitives, W::Array{Float64})
  # initialize
  TWe = fill(-Inf,prim.s_size)
  nde = zeros(Int64,prim.s_size)
  xe = zeros(Int64,prim.s_size)

  # find max value for each productivity level
  for s_index in 1:prim.s_size
  s = prim.s_vals[s_index]

  max_value = -Inf # initialize value

    for labor_choice in 1:prim.n_size
      n = prim.n_vals[labor_choice]

      # calculate exit value
      exit_value = prim.p*s*n^prim.theta - n

      # calculate value staying in market
      value = exit_value + prim.beta*dot(prim.F[s_index,:],W)

      if value >= exit_value
        if value > max_value
          max_value = value
          nde[s_index] = labor_choice
          xe[s_index] = 0
        end
      else
        if exit_value > max_value
          max_value = exit_value
          nde[s_index] = labor_choice
          xe[s_index] = 1
        end
      end
    end
    TWe[s_index] = max_value
  end
  TWe, nde, xe
end

## Value Function Iteration (note: only need to find fixed point of incumbent)

function vfi!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)
    if prim.beta == 0.0
        tol = Inf
    else
        tol = epsilon
    end

    for i in 1:max_iter
        # updates Tv and sigma in place
        res.TW, res.nd, res.x = bellman_operator_incumbent!(prim,res.W)

        # compute error and update the value with Tv inside results
        err = maxabs(res.TW .- res.W)
        copy!(res.W, res.TW)
        res.num_iter += 1

        if err < tol
            break
        end
    end

    res.TWe, res.nde, res.xe = bellman_operator_entrant!(prim,res.W)

    res
end

# ## Find Stationary distribution
#
# function create_statdist!(prim::Primitives, res::Results,
#   max_iter::Integer, epsilon::Real)
#
# N = prim.N
# m = prim.a_size
#
# #= Create transition matrix Tstar that is N x N, where N is the number
# of (a,s) combinations. Each row of Tstar is a conditional distribution over
# (a',s') conditional on the (a,s) today defined by row index =#
#
# Tstar = spzeros(N,N)
#
# for state_today in 1:N
#   choice_index = res.sigma[prim.a_s_indices[state_today,1],
#   prim.a_s_indices[state_today,2]] #a' given (a,s)
#   for state_tomorrow in 1:N
#     if prim.a_s_indices[state_tomorrow,1] == choice_index
#       Tstar[state_today,state_tomorrow] =
#         prim.markov[prim.a_s_indices[state_today,2]
#         ,prim.a_s_indices[state_tomorrow,2]]
#     end
#   end
# end
#
# #= Find stationary distribution. Start with any distribution over
# states and feed through Tstar matrix until convergence=#
#
# statdist = res.statdist
#
# num_iter = 0
#
#   for i in 1:max_iter
#
#       statdistprime = Tstar'*statdist
#
#       # compute error and update stationary distribution
#       err = maxabs(statdistprime .- statdist)
#       copy!(statdist, statdistprime)
#       num_iter += 1
#
#       if err < epsilon
#           break
#       end
#   end
#
#   res.statdist = statdist
#   res.num_dist = num_iter
#
#   res
#
# end
