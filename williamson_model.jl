#=
Program Name: williamson_model.jl
Creates Williamson (1998) model and utilities
=#

using QuantEcon: gridmake
#using ChebyshevApprox
#using Optim
using JuMP
using NLopt
#using Ipopt
#using Mosek
#using BlackBoxOptim

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
  cheby_unit_nodes :: Vector{Float64} ## Chebyshev nodes on [-1,1]
  chebyshev_flag::Int ## flag == 1 if using Chebyshev approximation
end

#= Outer Constructor for Primitives. Supplies default values, field
names, and creates grid objects =#

function Primitives(;beta::Float64=0.99, q::Float64=0.91, x::Float64=1.0,
  yH::Float64=1.6, yL::Float64=0.4, pi::Float64=0.5,
  ce::Float64=0.4713, w_size::Int64=5, chebyshev_flag::Int=1)

  # Calculate w_min and w_max

  w_min = pi*(1-exp(-(yH - yL)))
  w_max = pi*(1-exp(-(yH + x))) + (1-pi)*(1-exp(-(yL + x)))

  # Utility grid

  if chebyshev_flag == 1
    cheby_unit_nodes, w_vals = cheby_nodes!(w_min,w_max,w_size)
  else
    w_vals = linspace(w_min,w_max,w_size)
    cheby_unit_nodes = zeros(w_size)
  end

  primitives = Primitives(beta, q, x, yH, yL, pi, ce,
    w_min, w_max, w_size, w_vals, cheby_unit_nodes, chebyshev_flag)

  return primitives

end

## Type Results which holds results of the problem

type Results
    v::Array{Float64}
    Tv::Array{Float64}
    num_iter::Int
    wprime_indices::Array{Int,2}
    wprime_vals::Array{Float64,2}
    tau_vals::Array{Float64,2}
    statdist::Array{Float64}
    num_dist::Int
    principal_value::Array{Float64}
    consumption::Array{Float64,2}
    cheby_coeff::Vector{Float64}

    function Results(prim::Primitives)
        v = zeros(prim.w_size) # initialize cost with zeros
        wprime_indices = zeros(Int,(prim.w_size,2)) # initialize wprime choice index with zeros
        wprime_vals = zeros(prim.w_size,2) # initialize wprime choice with zeros
        statdist = ones(prim.w_size)*(1/(prim.w_size))# initialize stationary distribution with uniform
        cheby_coeff = zeros(4) # initialize Chebyshev coefficients with zeros
        res = new(v, similar(v), 0, wprime_indices, wprime_vals,
          similar(wprime_vals), statdist, 0, similar(v), similar(wprime_vals), cheby_coeff)

        res
    end

    # Version w/ initial v
    function Results(prim::Primitives,v::Array{Float64})
        wprime_indices = zeros(Int,(prim.w_size,2))
        wprime_vals = zeros(prim.w_size,2)
        statdist = ones(prim.w_size)*(1/(prim.w_size))
        cheby_coeff = zeros(4)
        res = new(v, similar(v), 0, wprime_indices, wprime_vals,
          similar(wprime_vals), statdist, 0, similar(v), similar(wprime_vals), cheby_coeff)

        res
    end
end

## Solve Program

#Without initial value function (will be initialized at zeros)
function SolveProgram(bellman::Function,prim::Primitives;
  max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
    res = Results(prim)
    vfi!(bellman,prim, res, max_iter_vfi, epsilon_vfi)
    create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
    res
end

# With Initial value function
function SolveProgram(bellman::Function,prim::Primitives, v::Array{Float64};
  max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
    res = Results(prim, v)
    vfi!(bellman,prim, res, max_iter_vfi, epsilon_vfi)
    create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
    res
end

#= Internal Utilities =#

## Chebyshev approximation (order 3)

# find Chebyshev nodes

function cheby_nodes!(lower::Float64,upper::Float64,node_size::Int64)

  nodes_unit = zeros(node_size)
  nodes_adjust = zeros(node_size)

  # compute nodes on [-1,1]
  for k in 1:node_size
    nodes_unit[k] = -cos(((2*k-1)/(2*node_size))*pi)
  end

  # adjust nodes to [lower,upper]
  for k in 1:node_size
    nodes_adjust[k] = (nodes_unit[k]+1)*((upper-lower)/2) + lower
  end

  return nodes_unit, nodes_adjust
end

# find Chebyshev coefficients (order 3)

function cheby_coeff_three!(func_image::Array,nodes_unit::Array)

  node_size = length(nodes_unit)
  cheby_coeff = zeros(4)

  numerators = zeros(4)
  denominators = zeros(4)

  for k in 1:node_size
    numerators[1] += func_image[k]*1
    denominators[1] += 1^2
    numerators[2] += func_image[k]*nodes_unit[k]
    denominators[2] += nodes_unit[k]^2
    numerators[3] += func_image[k]*(2*nodes_unit[k]^2-1)
    denominators[3] += (2*nodes_unit[k]^2-1)^2
    numerators[4] += func_image[k]*(4*nodes_unit[k]^3-3*nodes_unit[k])
    denominators[4] += (4*nodes_unit[k]^3-3*nodes_unit[k])^2
  end

  cheby_coeff = numerators./denominators

  return cheby_coeff
end

# Chebyshev approximation (order 3)

function cheby_approx!(x,cheby_coeff::Array,lower,upper)

  fhat = cheby_coeff[1]*1 +
    cheby_coeff[2]*(2*(x-lower)/(upper-lower) - 1) +
    cheby_coeff[3]*(2*(2*(x-lower)/(upper-lower) - 1)^2 - 1) +
    cheby_coeff[4]*(4*(2*(x-lower)/(upper-lower) - 1)^3 -
      3*(2*(x-lower)/(upper-lower) - 1))

  return fhat
end

## Bellman Operators

# Grid Search

function bellman_gridsearch!(prim::Primitives, v::Array{Float64})
  # initialize
  Tv = fill(Inf,prim.w_size)
  wprime_indices = zeros(Int,(prim.w_size,2))
  wprime_vals = zeros(prim.w_size,2)
  tau_vals = zeros(prim.w_size,2)

  # loop over promised utility values
  for w_index in 1:prim.w_size
    w = prim.w_vals[w_index]

    # initialize cost for (wprimeH,wprimeL) combinations
    min_cost = Inf

    # find min cost for each (wprimeH,wprimeL) combination
    for wprimeH_index in 1:prim.w_size
      wprimeH = prim.w_vals[wprimeH_index]
      for wprimeL_index in 1:prim.w_size
        wprimeL = prim.w_vals[wprimeL_index]

        # find transfer schedule determined by promise constraint and binding IC
        # check that (tauH,tauL) is defined and within constraints
        if ((wprimeL-1)*prim.beta - w + 1)/
        (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1)) > 0 &&
        ((1-prim.pi)*prim.beta*(wprimeH-wprimeL)*exp(prim.yH-prim.yL) +
          prim.beta*(prim.pi*wprimeH + (1-prim.pi)*wprimeL) - w + 1 - prim.beta)/
        (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1)) > 0 &&
        -prim.yH <= -prim.yH -
          log(
            ((wprimeL-1)*prim.beta - w + 1)/
            (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1))
            ) <= prim.x &&
        -prim.yL <= -prim.yH -
          log(
            ((1-prim.pi)*prim.beta*(wprimeH-wprimeL)*exp(prim.yH-prim.yL) +
              prim.beta*(prim.pi*wprimeH + (1-prim.pi)*wprimeL) - w + 1 - prim.beta)/
            (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1))
            ) <= prim.x

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
            wprime_vals[w_index,1] = prim.w_vals[wprimeH_index]
            wprime_vals[w_index,2] = prim.w_vals[wprimeL_index]
            tau_vals[w_index,1] = tauH
            tau_vals[w_index,2] = tauL
          end

        end

      end
    end
    Tv[w_index] = min_cost
  end
  Tv, wprime_indices, wprime_vals, tau_vals
end

# Interpolation with Chebyshev Polynomials

# JuMP

# define cost function with Chebyshev approximation and 1/1-w term to improve fit

# cost(wprimeH,wprimeL,tauH,tauL,w,q,pi,w_min,w_max,cheby_coeff1,cheby_coeff2,cheby_coeff3,cheby_coeff4) =
#   (1-q)*(pi*tauH + (1-pi)*tauL) +
#   q*(pi*
#     (cheby_coeff1 + cheby_coeff2*(2*(wprimeH-w_min)/(w_max-w_min) - 1) +
#     cheby_coeff3*(2*(2*(wprimeH-w_min)/(w_max-w_min) - 1)^2 - 1) +
#     cheby_coeff4*(4*(2*(wprimeH-w_min)/(w_max-w_min) - 1)^3 -
#       3*(2*(wprimeH-w_min)/(w_max-w_min) - 1))) +
#   (1-pi)*
#   (cheby_coeff1 + cheby_coeff2*(2*(wprimeL-w_min)/(w_max-w_min) - 1) +
#   cheby_coeff3*(2*(2*(wprimeL-w_min)/(w_max-w_min) - 1)^2 - 1) +
#   cheby_coeff4*(4*(2*(wprimeL-w_min)/(w_max-w_min) - 1)^3 -
#     3*(2*(wprimeL-w_min)/(w_max-w_min) - 1)))
#   ) - 1/(1-w)
#
# JuMP.register(:cost,13,cost,autodiff=true)

function bellman_chebyshev_jump!(prim::Primitives, v::Array{Float64})
  # initialize
  Tv = fill(Inf,prim.w_size)
  wprime_indices = zeros(Int,(prim.w_size,2))
  wprime_vals = zeros(prim.w_size,2)
  tau_vals = zeros(prim.w_size,2)

  # find Chebyshev coefficients
  cheby_coeff = cheby_coeff_three!(v,prim.cheby_unit_nodes)

  # construct model for JuMP
  m = Model(solver=NLoptSolver(algorithm=:LD_SLSQP))

  # choice variables
  @variable(m, prim.w_min <= wprimeH <= prim.w_max, start = (prim.w_min + prim.w_max)/2)
  @variable(m, prim.w_min <= wprimeL <= prim.w_max, start = (prim.w_min + prim.w_max)/2)
  @variable(m, -prim.yH <= tauH <= prim.x, start = prim.x)
  @variable(m, -prim.yL <= tauL <= prim.x, start = prim.x)

  # set promised utility parameter to first value
  @NLparameter(m, w == prim.w_vals[1])
  #@NLparameter(m, q == prim.q)
  #@NLparameter(m, pi == prim.pi)
  #@NLparameter(m, w_min == prim.w_min)
  #@NLparameter(m, w_max == prim.w_max)
  #@NLparameter(m, cheby_coeff1 == cheby_coeff[1])
  #@NLparameter(m, cheby_coeff2 == cheby_coeff[2])
  #@NLparameter(m, cheby_coeff3 == cheby_coeff[3])
  #@NLparameter(m, cheby_coeff4 == cheby_coeff[4])

  # # objective function
  # @NLobjective(m, Min, cost(wprimeH,wprimeL,tauH,tauL,w,
  #   q,pi,w_min,w_max,cheby_coeff1,cheby_coeff2,cheby_coeff3,cheby_coeff4))

  # objective function
  @NLobjective(m, Min,
  (1-prim.q)*(prim.pi*tauH + (1-prim.pi)*tauL) +
  prim.q*(prim.pi*
    (cheby_coeff[1] + cheby_coeff[2]*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1) +
    cheby_coeff[3]*(2*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1)^2 - 1) +
    cheby_coeff[4]*(4*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1)^3 -
      3*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1))) +
  (1-prim.pi)*
  (cheby_coeff[1] + cheby_coeff[2]*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1) +
  cheby_coeff[3]*(2*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1)^2 - 1) +
  cheby_coeff[4]*(4*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1)^3 -
    3*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1)))
  ) - prim.q/(1-w)
  )

  # @NLobjective(m, Max,
  # -(1-prim.q)*(prim.pi*tauH + (1-prim.pi)*tauL) +
  # prim.q*(prim.pi*
  #   (cheby_coeff[1] + cheby_coeff[2]*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1) +
  #   cheby_coeff[3]*(2*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1)^2 - 1) +
  #   cheby_coeff[4]*(4*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1)^3 -
  #     3*(2*(wprimeH-prim.w_min)/(prim.w_max-prim.w_min) - 1))) +
  # (1-prim.pi)*
  # (cheby_coeff[1] + cheby_coeff[2]*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1) +
  # cheby_coeff[3]*(2*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1)^2 - 1) +
  # cheby_coeff[4]*(4*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1)^3 -
  #   3*(2*(wprimeL-prim.w_min)/(prim.w_max-prim.w_min) - 1)))
  # ) + (1-prim.q)*prim.x + prim.q/(1-w)
  # )

  # constraints

  # full constraints
  @NLconstraint(m, w == (1-prim.beta)*(prim.pi*(1-exp(-prim.yH - tauH)) +
    (1-prim.pi)*(1-exp(-prim.yL - tauL))) +
    prim.beta*(prim.pi*wprimeH + (1-prim.pi)*wprimeL))
  @NLconstraint(m, (1-prim.beta)*(1-exp(-prim.yH - tauL)) + prim.beta*wprimeL <=
    (1-prim.beta)*(1-exp(-prim.yH - tauH)) + prim.beta*wprimeH)
  @NLconstraint(m, (1-prim.beta)*(1-exp(-prim.yL - tauH)) + prim.beta*wprimeH <=
    (1-prim.beta)*(1-exp(-prim.yL - tauL)) + prim.beta*wprimeL)

  # impose binding IC and ignore slack IC
  # @NLconstraint(m, w == (1-prim.beta)*(prim.pi*(1-exp(-prim.yH - tauH)) +
  #   (1-prim.pi)*(1-exp(-prim.yL - tauL))) +
  #   prim.beta*(prim.pi*wprimeH + (1-prim.pi)*wprimeL))
  # @NLconstraint(m, (1-prim.beta)*(1-exp(-prim.yH - tauL)) + prim.beta*wprimeL ==
  #   (1-prim.beta)*(1-exp(-prim.yH - tauH)) + prim.beta*wprimeH)

  # loop over promised utility values
  for w_index in 1:prim.w_size

    # update promised utility parameter (redundant in first iteration)
    setvalue(w, prim.w_vals[w_index])

    # solve problem given promised utility
    solve(m)

    # store results
    Tv[w_index] = getobjectivevalue(m)
    wprime_vals[w_index,1] = getvalue(wprimeH)
    wprime_vals[w_index,2] = getvalue(wprimeL)
    tau_vals[w_index,1] = getvalue(tauH)
    tau_vals[w_index,2] = getvalue(tauL)

  end
  #Tv, wprime_indices, wprime_vals, tau_vals
  Tv, wprime_vals, tau_vals, cheby_coeff

end

tic()
testcheb = bellman_chebyshev_jump!(prim,v)
toc()

# Optim, runs into trouble with log domain error

# function bellman_chebyshev_optim!(prim::Primitives, v::Array{Float64})
#   # initialize
#   Tv = fill(Inf,prim.w_size)
#   #wprime_indices = zeros(Int,(prim.w_size,2))
#   wprime_vals = zeros(prim.w_size,2)
#   tau_vals = zeros(prim.w_size,2)
#
#   # find Chebyshev coefficients
#   cheby_coeff = cheby_coeff_three!(v,prim.cheby_unit_nodes)
#
#   # loop over promised utility values
#   for w_index in 1:prim.w_size
#     w = prim.w_vals[w_index]
#
#     #= find min cost given promised utility w combination and
#     binding IC constraint =#
#
#     tauL(wprime) = -prim.yH -
#       log(
#         ((wprime[2]-1)*prim.beta - w + 1)/
#         (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1))
#         )
#     tauH(wprime) = -prim.yH -
#       log(
#         ((1-prim.pi)*prim.beta*(wprime[1]-wprime[2])*exp(prim.yH-prim.yL) +
#           prim.beta*(prim.pi*wprime[1] + (1-prim.pi)*wprime[2]) - w + 1 - prim.beta)/
#         (((prim.pi-1)*exp(prim.yH-prim.yL)-prim.pi)*(prim.beta-1))
#         )
#
#       # define cost function with Chebyshev approximation
#       cost(wprime) = (1-prim.q)*(prim.pi*tauH(wprime) + (1-prim.pi)*tauL(wprime)) +
#         prim.q*(prim.pi*cheby_approx!(wprime[1],cheby_coeff,prim.w_min,prim.w_max) +
#         (1-prim.pi)*cheby_approx!(wprime[2],cheby_coeff,prim.w_min,prim.w_max)) -
#         1/(1-w)
#
#     lower = [prim.w_min, prim.w_min]
#     upper = [prim.w_max, prim.w_max]
#     initial = upper - [1e-6, 1e-6]
#     opt_results = optimize(DifferentiableFunction(cost),initial,lower,upper,Fminbox(),optimizer=GradientDescent)
#
#     wprime_vals[w_index,1] = opt_results.minimizer[1]
#     wprime_vals[w_index,2] = opt_results.minimizer[2]
#     tau_vals[w_index,1] = tauH(opt_results.minimizer)
#     tau_vals[w_index,2] = tauL(opt_results.minimizer)
#
#     # Find inidices?????
#
#   end
#   #Tv, wprime_indices, wprime_vals, tau_vals
#   Tv, wprime_vals, tau_vals
# end
#

## Value Function Iteration

function vfi!(bellman::Function,prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)
    if prim.beta == 0.0
        tol = Inf
    else
        tol = epsilon
    end

    res.num_iter = 0

    for i in 1:max_iter
        # updates Tv and choice arrays in place
        if bellman == bellman_gridsearch!
          res.Tv, res.wprime_indices, res.wprime_vals, res.tau_vals = bellman(prim,res.v)
        elseif bellman == bellman_chebyshev_jump!
          res.Tv, res.wprime_vals, res.tau_vals, res.cheby_coeff = bellman(prim,res.v)
        else
          msg = "Use either Grid Search or JuMP NL Solver"
          throw(ArgumentError(msg))
        end

        # compute error and update the value with Tv inside results
        err = maxabs(res.Tv .- res.v)
        copy!(res.v, res.Tv)
        res.num_iter += 1
        println("Iter: ", i, " Error: ", err)

        if err < tol && bellman == bellman_gridsearch!
          break
        elseif err < tol && bellman == bellman_chebyshev_jump!
          #interp decision rules here
          break
        end
    end

    res
end

## Find Stationary distribution

function create_statdist!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)

# number of states (w,y) is 2 times w grid size
N = 2*prim.w_size

#= Create transition matrix Tstar that is N x N, where N is the number
of (promised utility,endowment) combinations. Each row of Tstar is a
distribution over (w',y') conditional on (w,y) today defined by row index =#

Tstar = spzeros(N,N)

for state_today in 1:N

  # indentify utility and endowment indices from state index
  if state_today <= prim.w_size # endowed with yH
    w_index = state_today # w index
    endow_index = 1 # yH index
  else # endowed with yL
    w_index = state_today - prim.w_size # w index
    endow_index = 2 # yL index
  end

  # use decision rule to determine w' given (w,y)
  wprime_index = res.wprime_indices[w_index,endow_index]

  # fill in transition matrix using endowment process (iid in this case)
  for state_tomorrow in 1:N

    if state_tomorrow <= prim.w_size # endowment tomorrow is yH
      if state_tomorrow == wprime_index
        Tstar[state_today,state_tomorrow] = prim.pi
      end
    else # endowment tomorrow is yL
      if state_tomorrow - prim.w_size == wprime_index
        Tstar[state_today,state_tomorrow] = (1-prim.pi)
      end
    end

  end

end

#= Find stationary distribution. Start with uniform distribution over
states and feed through Tstar matrix until convergence=#

tempdist = ones(N)*(1/N)

res.num_dist = 0

  for i in 1:max_iter

      tempdistprime = Tstar'*tempdist

      # compute error and update stationary distribution
      err = maxabs(tempdistprime .- tempdist)
      copy!(tempdist, tempdistprime)
      res.num_dist += 1

      if err < epsilon
        break
      end
  end

  # collapse distribution over (w,y) into distribution over w only

  for w_index in 1:prim.w_size
    res.statdist[w_index] = tempdist[w_index] + tempdist[w_index + prim.w_size]
  end

  res

end

## Calculate Principal Value and Agent Consumption

function value_consumption!(prim::Primitives,res::Results)

  res.principal_value = -res.v - (1/prim.q)*(1-prim.q)*prim.x
  res.consumption = res.tau_vals + [prim.yH prim.yL].*ones(res.tau_vals)

  res
end
