#=
Program Name: williamson_model.jl
Creates Williamson (1998) Model and Utilities
=#

using QuantEcon: gridmake

## Create composite type to hold model primitives

type Primitives
  beta :: Float64 ## agent discount rate
  q :: Float64 ## discount bond price
  x :: Float64 ## principal endowment
  yH :: Float64 ## high agent endowment
  yL :: Float64 ## low agent endowment
  pi :: Float64 ## probability of high endowment (iid across time)
  ce :: Float64 ## fixed cost for principal
  w_min :: Float64 ## minimum incentive compatible expected utility
  w_max :: Float64 ## maximum feasible expected utility
  w_size :: Int64 ## size of expected utility grid
  w_vals :: Vector{Float64} ## expected utility grid
  #w_indices :: Array{Int64} ## indices of expected utility grid
  #tau_min :: Float64 ## minimum transfer value
  #tau_max :: Float64 ## maximum transfer value
  #tau_size :: Int64 ## size of transfer value grid
  #tau_vals :: Vector{Float64} ## transfer value grid
  #tau_indices :: Array{Int64} ## indices of transfer value grid
end

#= Outer Constructor for Primitives. Supplies default values, field
names, and creates grid objects =#

function Primitives(;beta::Float64=0.9, q::Float64=0.91, x::Float64=1.0,
  yH::Float64=1.6, yL::Float64=0.4, pi::Float64=0.5,
  ce::Float64=0.4713, w_size::Int64=100, tau_min::Float64=-2.0,
  tau_max::Float64=1.0, tau_size::Int64=100)

  # Calculate w_min and w_max

  w_min = pi*(1-exp(-(yH - yL)))
  w_max = pi*(1-exp(-(yH + x))) + (1-pi)*(1-exp(-(yL + x)))

  # Grids

  w_vals = linspace(w_min, w_max, w_size)
  w_indices = gridmake(1:w_size)

  tau_vals = linspace(tau_min, tau_max, tau_size)
  tau_indices = gridmake(1:tau_size)

  primitives = Primitives(beta, q, x, yH, yL, pi, ce,
    w_min, w_max, w_size, w_vals, w_indices,
    tau_min, tau_max, tau_size, tau_vals, tau_indices)

  return primitives

end

## Type Results which holds results of the problem

type Results
    v::Array{Float64}
    Tv::Array{Float64}
    num_iter::Int
    wprime_indices::Array{Int,2}
    wprime::Array{Float64,2}
    tau::Array{Float64,2}
    statdist::Array{Float64}
    num_dist::Int

    function Results(prim::Primitives)
        v = zeros(prim.w_size) # initialize cost with zeros
        wprime_indices = zeros(Int,(prim.w_size,2)) # initialize wprime choice index with zeros
        wprime = zeros(prim.w_size,2) # initialize wprime choice with zeros
        statdist = ones(prim.w_size)*(1/prim.w_size)# initialize stationary distribution with uniform
        res = new(v, similar(v), 0, wprime_indices, wprime,
          similar(wprime), statdist, 0)

        res
    end

    # Version w/ initial v
    function Results(prim::Primitives,v::Array{Float64})
        wprime_indices = zeros(Int,(prim.w_size,2))
        wprime = zeros(prim.w_size,2)
        statdist = ones(prim.w_size)*(1/prim.w_size)
        res = new(v, similar(v), 0, wprime_indices, wprime,
          similar(wprime), statdist, 0)

        res
    end
end

# ## Solve Discrete Dynamic Program
#
# # Without initial value function (will be initialized at zeros)
# function SolveProgram(prim::Primitives;
#   max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3,
#   max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
#     res = Results(prim)
#     vfi!(prim, res, max_iter_vfi, epsilon_vfi)
#     create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
#     res
# end
#
# # With Initial value function
# function SolveProgram(prim::Primitives, v::Array{Float64,2};
#   max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3,
#   max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
#     res = Results(prim, v)
#     vfi!(prim, res, max_iter_vfi, epsilon_vfi)
#     create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
#     res
# end

#= Internal Utilities =#

## Bellman Operator

function bellman_operator!(prim::Primitives, v::Array{Float64})
  # initialize
  Tv = fill(Inf,prim.w_size)
  #Tv = fill(1000.0,prim.w_size)
  wprime_indices = zeros(Int,(prim.w_size,2))
  wprime = zeros(prim.w_size,2)
  tau = zeros(prim.w_size,2)

  # loop over promised utility values
  for w_index in 1:prim.w_size
    w = prim.w_vals[w_index]

    # initialize cost for (wprimeH,wprimeL) combinations
    min_cost = Inf
    #min_cost = 1000.0

    # find min cost for each (wprimeH,wprimeL) combination
    for wprimeH_index in 1:prim.w_size
      wprimeH = prim.w_vals[wprimeH_index]
      for wprimeL_index in 1:prim.w_size
        wprimeL = prim.w_vals[wprimeL_index]

        # find transfer schedule determined by promise constraint and binding IC
        # check that (tauH,tauL) is defined
        if ((wprimeL-1)*prim.beta - w + 1)/
        (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1)) > 0 &&
        ((1-prim.pi)*prim.beta*(wprimeH-wprimeL)*exp(prim.yH-prim.yL) +
          prim.beta*(prim.pi*wprimeH + (1-prim.pi)*wprimeL) - w + 1 - prim.beta)/
        (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1)) > 0

          tauL = -prim.yH -
            log(
              ((wprimeL-1)*prim.beta - w + 1)/
              (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1))
              )
          tauH = -prim.yH -
            log(
              ((1-prim.pi)*prim.beta*(wprimeH-wprimeL)*exp(prim.yH-prim.yL) +
                prim.beta*(prim.pi*wprimeH + (1-prim.pi)*wprimeL) - w + 1 - prim.beta)/
              (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1))
              )

          # calculate cost given wprimeH, wprimeL, tauH, tauL
          cost = (1-prim.q)*(prim.pi*tauH + (1-prim.pi)*tauL) +
            prim.q*(prim.pi*v[wprimeH_index] + (1-prim.pi)*v[wprimeL_index])

          if cost < min_cost
            min_cost = cost
            wprime_indices[w_index,1] = wprimeH_index
            wprime_indices[w_index,2] = wprimeL_index
            wprime[w_index,1] = prim.w_vals[wprimeH_index]
            wprime[w_index,2] = prim.w_vals[wprimeL_index]
            tau[w_index,1] = tauH
            tau[w_index,2] = tauL
          end

        end #taucheck

      end #wprimeL
    end #wprimeH
    Tv[w_index] = min_cost
  end #w
  Tv, wprime_indices, wprime, tau
end

## Value Function Iteration

function vfi!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)
    if prim.beta == 0.0
        tol = Inf
    else
        tol = epsilon
    end

    res.num_iter = 0

    for i in 1:max_iter
        # updates Tv and choice arrays in place
        res.Tv, res.wprime, res.tau = bellman_operator!(prim,res.v)

        # compute error and update the value with Tv inside results
        err = maxabs(res.Tv .- res.v)
        copy!(res.v, res.Tv)
        res.num_iter += 1
        println("Iter: ", i, " Error: ", err)

        if err < tol
            break
        end
    end

    res
end

## Find Stationary distribution

function create_statdist!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)

N = 2*prim.w_size

#= Create transition matrix Tstar that is N x N, where N is the number
of promised utility/endowment combinations. Each row of Tstar is a conditional
distribution over w' conditional on (w,y) today defined by row index =#

Tstar = spzeros(N,N)

for w_index in 1:N
  choice_index = res.sigma[prim.a_s_indices[state_today,1],
  prim.a_s_indices[state_today,2]] #a' given (a,s)
  for state_tomorrow in 1:N
    if prim.a_s_indices[state_tomorrow,1] == choice_index
      Tstar[state_today,state_tomorrow] =
        prim.markov[prim.a_s_indices[state_today,2]
        ,prim.a_s_indices[state_tomorrow,2]]
    end
  end
end

#= Find stationary distribution. Start with any distribution over
states and feed through Tstar matrix until convergence=#

statdist = res.statdist

num_iter = 0

  for i in 1:max_iter

      statdistprime = Tstar'*statdist

      # compute error and update stationary distribution
      err = maxabs(statdistprime .- statdist)
      copy!(statdist, statdistprime)
      num_iter += 1

      if err < epsilon
          break
      end
  end

  res.statdist = statdist
  res.num_dist = num_iter

  res

end
