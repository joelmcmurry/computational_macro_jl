#=
Program Name: williamson_model.jl
Creates Williamson (1998) model and utilities
=#

using QuantEcon: gridmake
using JuMP
using NLopt
using Interpolations

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
  w_size_itp :: Int64 ## size of interpolated decision rule grid
  w_vals :: Vector{Float64} ## expected utility grid
  w_vals_itp :: Vector{Float64} ## expected utility grid size of interpolated grids
  cheby_unit_nodes :: Vector{Float64} ## Chebyshev nodes on [-1,1]
end

#= Outer Constructor for Primitives. Supplies default values, field
names, and creates grid objects =#

function Primitives(;beta::Float64=0.89, q::Float64=0.91, x::Float64=1.0,
  yH::Float64=1.6, yL::Float64=0.4, pi::Float64=0.5,
  ce::Float64=0.4713, w_size::Int64=5, w_size_itp::Int64=500)

  # Calculate w_min and w_max

  w_min = pi*(1-exp(-(yH - yL)))
  w_max = pi*(1-exp(-(yH + x))) + (1-pi)*(1-exp(-(yL + x)))

  # Utility grids

  cheby_unit_nodes, w_vals = cheby_nodes!(w_min,w_max,w_size)
  w_vals_itp = linspace(w_min,w_max,w_size_itp)

  primitives = Primitives(beta, q, x, yH, yL, pi, ce,
    w_min, w_max, w_size, w_size_itp, w_vals, w_vals_itp, cheby_unit_nodes)

  return primitives

end

## Type Results which holds results of the problem

type Results
    v::Array{Float64}
    Tv::Array{Float64}
    num_iter::Int
    wprime_vals::Array{Float64,2}
    wprime_indices_itp::Array{Int,2}
    tau_vals::Array{Float64,2}
    tau_vals_itp::Array{Float64,2}
    statdist::Array{Float64}
    num_dist::Int
    consumption::Array{Float64,2}
    cheby_coeff::Vector{Float64}

    function Results(prim::Primitives)
        v = zeros(prim.w_size) # initialize cost with zeros
        wprime_vals = zeros(prim.w_size,2) # initialize wprime choice with zeros
        wprime_indices_itp = zeros(Int,(prim.w_size_itp,2)) # initialize wprime choice index with zeros
        statdist = ones(prim.w_size_itp)*(1/(prim.w_size_itp))# initialize stationary distribution with uniform
        cheby_coeff = zeros(4) # initialize Chebyshev coefficients with zeros
        res = new(v, similar(v), 0, wprime_vals, wprime_indices_itp,
          similar(wprime_vals), similar(wprime_indices_itp,Float64), statdist, 0,
          similar(wprime_vals), cheby_coeff)

        res
    end

    # Version w/ initial v
    function Results(prim::Primitives,v::Array{Float64})
      wprime_vals = zeros(prim.w_size,2)
      wprime_indices_itp = zeros(Int,(prim.w_size_itp,2))
      statdist = ones(prim.w_size_itp)*(1/(prim.w_size_itp))
      cheby_coeff = zeros(4)
      res = new(v, similar(v), 0, wprime_vals, wprime_indices_itp,
        similar(wprime_vals), similar(wprime_indices_itp,Float64), statdist, 0,
        similar(wprime_vals), cheby_coeff)

        res
    end
end

## Solve Program

# Without initial value function (will be initialized at zeros)
function SolveProgram(prim::Primitives;
  max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
    res = Results(prim)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi)
    create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
    res
end

# With Initial value function
function SolveProgram(prim::Primitives, v::Array{Float64};
  max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
    res = Results(prim, v)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi)
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

## Bellman Operator

function bellman_chebyshev_jump!(prim::Primitives, v::Array{Float64})
  # initialize
  Tv = fill(Inf,prim.w_size)
  wprime_vals = zeros(prim.w_size,2)
  tau_vals = zeros(prim.w_size,2)

  # find Chebyshev coefficients
  cheby_coeff = cheby_coeff_three!(v,prim.cheby_unit_nodes)

  # loop over promised utility values
  for w_index in 1:prim.w_size

    # construct model for JuMP
    m = Model(solver=NLoptSolver(algorithm=:LD_SLSQP))

    # choice variables
    @variable(m, prim.w_min <= wprimeH <= prim.w_max, start = (prim.w_min + prim.w_max)/2)
    @variable(m, prim.w_min <= wprimeL <= prim.w_max, start = (prim.w_min + prim.w_max)/2)
    @variable(m, -prim.yH <= tauH <= prim.x, start = prim.x)
    @variable(m, -prim.yL <= tauL <= prim.x, start = prim.x)

    # promised utility
    @NLparameter(m, w == prim.w_vals[w_index])

    # objective function
    @NLobjective(m, Max,
    (1-prim.q)*(prim.pi*(prim.x-tauH) + (1-prim.pi)*(prim.x-tauL)) +
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
     )
    )

    # constraints
    @NLconstraint(m, w == (1-prim.beta)*(prim.pi*(1-exp(-prim.yH - tauH)) +
      (1-prim.pi)*(1-exp(-prim.yL - tauL))) +
      prim.beta*(prim.pi*wprimeH + (1-prim.pi)*wprimeL))
    @NLconstraint(m, (1-prim.beta)*(1-exp(-prim.yH - tauL)) + prim.beta*wprimeL <=
      (1-prim.beta)*(1-exp(-prim.yH - tauH)) + prim.beta*wprimeH)
    @NLconstraint(m, (1-prim.beta)*(1-exp(-prim.yL - tauH)) + prim.beta*wprimeH <=
      (1-prim.beta)*(1-exp(-prim.yL - tauL)) + prim.beta*wprimeL)

    # solve problem given promised utility
    solve(m)

    # store results
    Tv[w_index] = getobjectivevalue(m)
    wprime_vals[w_index,1] = getvalue(wprimeH)
    wprime_vals[w_index,2] = getvalue(wprimeL)
    tau_vals[w_index,1] = getvalue(tauH)
    tau_vals[w_index,2] = getvalue(tauL)

  end
  Tv, wprime_vals, tau_vals, cheby_coeff

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
        res.Tv, res.wprime_vals, res.tau_vals, res.cheby_coeff = bellman_chebyshev_jump!(prim,res.v)

        # compute error and update the value with Tv inside results
        err = maxabs(res.Tv .- res.v)
        copy!(res.v, res.Tv)
        res.num_iter += 1
        #println("Iter: ", i, " Error: ", err)

        if err < tol
          res.wprime_indices_itp, res.tau_vals_itp = interp_decision!(prim,res)
          break
        end
    end

    res
end

## Interpolate decision rules for wprimeH, wprimeL

function interp_decision!(prim::Primitives,res::Results)

    grid_size = prim.w_size_itp

    w_grid = prim.w_vals_itp

    # evenly space "grid_size" number of points between 1 and "w_size"
    grid_to_itp = linspace(1,prim.w_size,grid_size)

    itp_wprimeH = interpolate(res.wprime_vals[:,1],BSpline(Linear()),OnGrid())
    itp_wprimeL = interpolate(res.wprime_vals[:,2],BSpline(Linear()),OnGrid())

    itp_tauH_vals = interpolate(res.tau_vals[:,1],BSpline(Linear()),OnGrid())
    itp_tauL_vals= interpolate(res.tau_vals[:,2],BSpline(Linear()),OnGrid())

    wprime_indices_itp = zeros(grid_size,2)
    tau_vals_itp = zeros(grid_size,2)

    # wprimeH
    for domain_index in 1:grid_size
      mindist = Inf
      match_index = 0

      # convert grid index to interpolation index
      itp_index = grid_to_itp[domain_index]

      for search_index in 1:grid_size
        dist = abs(itp_wprimeH[itp_index] - w_grid[search_index])
        if dist < mindist
          mindist = dist
          match_index = search_index
        end
      end
      wprime_indices_itp[domain_index,1] = match_index
      tau_vals_itp[domain_index,1] = itp_tauH_vals[itp_index]
    end

    # wprimeL
    for domain_index in 1:grid_size
      mindist = Inf
      match_index = 0

      # convert grid index to interpolation index
      itp_index = grid_to_itp[domain_index]

      for search_index in 1:grid_size
        dist = abs(itp_wprimeL[itp_index] - w_grid[search_index])
        if dist < mindist
          mindist = dist
          match_index = search_index
        end
      end
      wprime_indices_itp[domain_index,2] = match_index
      tau_vals_itp[domain_index,2] = itp_tauL_vals[itp_index]
    end

  wprime_indices_itp, tau_vals_itp
end

## Find Stationary distribution

function create_statdist!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)

# number of states (w,y) is 2 times grid size used in decision rule interpolation
grid_size = prim.w_size_itp
N = 2*grid_size

#= Create transition matrix Tstar that is N x N, where N is the number
of (promised utility,endowment) combinations. Each row of Tstar is a
distribution over (w',y') conditional on (w,y) today defined by row index =#

Tstar = spzeros(N,N)

for state_today in 1:N

  # indentify utility and endowment indices from state index
  if state_today <= grid_size # endowed with yH
    w_index = state_today # w index
    endow_index = 1 # yH index
  else # endowed with yL
    w_index = state_today - grid_size # w index
    endow_index = 2 # yL index
  end

  # use decision rule to determine w' given (w,y)
  wprime_index = res.wprime_indices_itp[w_index,endow_index]

  # fill in transition matrix using endowment process (iid in this case)
  for state_tomorrow in 1:N

    if state_tomorrow <= grid_size # endowment tomorrow is yH
      if state_tomorrow == wprime_index
        Tstar[state_today,state_tomorrow] = prim.pi
      end
    else # endowment tomorrow is yL
      if state_tomorrow - grid_size == wprime_index
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

  for w_index in 1:grid_size
    res.statdist[w_index] = tempdist[w_index] + tempdist[w_index + grid_size]
  end

  res

end

## Calculate Agent Consumption

function consumption!(prim::Primitives,res::Results)

  res.consumption = res.tau_vals + [prim.yH prim.yL].*ones(res.tau_vals)

  res
end
