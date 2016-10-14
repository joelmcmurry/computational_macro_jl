#=Huggett Model and Utilities=#

using QuantEcon: gridmake

## Create compostive type to hold model primitives

type Primitives
  beta :: Float64 ## discount rate
  alpha :: Float64 ## risk aversion
  q :: Float64 ## interest rate
  a_min :: Float64 ## minimum asset value
  a_max :: Float64 ## maximum asset value
  a_size :: Int64 ## size of asset grid
  a_vals :: Vector{Float64} ## asset grid
  a_indices :: Array{Int64} ## indices of choices
  s_size :: Int64 ## number of earnings values
  s_vals:: Array{Float64,1} ## values of employment states
  markov :: Array{Float64,2} ## Markov process for earnings
  a_s_vals :: Array{Float64} ## array of states: (a,s) combinations
  a_s_indices :: Array{Int64} ## indices of states: (a,s) combinations
  N :: Int64 ## number of possible (a,s) combinations
end

#= Outer Constructor for Primitives. Supplies default values, field
names, and creates grid objects=#

function Primitives(;beta::Float64=0.9932, alpha::Float64=1.5,
  q::Float64=0.9, a_min::Float64=-2.0, a_max::Float64=5.0,
  a_size::Int64=100, markov=[0.97 (1-0.97);0.5 (1-0.5)],
  s_vals = [1, 0.5])

  # Grids

  a_vals = linspace(a_min, a_max, a_size)
  s_size = length(markov[1:2,1])
  a_indices = gridmake(1:a_size)
  N = a_size*s_size
  a_s_vals = gridmake(a_vals,s_vals)
  a_s_indices = gridmake(1:a_size,1:s_size)

  primitives = Primitives(beta, alpha, q, a_min, a_max, a_size,
  a_vals, a_indices, s_size, s_vals, markov, a_s_vals, a_s_indices, N)

  return primitives

end

## Type Results which holds results of the problem

type Results
    v::Array{Float64,2}
    Tv::Array{Float64,2}
    num_iter::Int
    sigma::Array{Int,2}
    statdist::Array{Float64}
    num_dist::Int

    function Results(prim::Primitives)
        v = zeros(prim.a_size,prim.s_size) # Initialise value with zeroes
        statdist = ones(prim.N)*(1/prim.N)# Initialize stationary distribution with uniform
        res = new(v, similar(v), 0, similar(v, Int), statdist,0)

        res
    end

    # Version w/ initial v
    function Results(prim::Primitives,v::Array{Float64,2})
        statdist = ones(prim.N)*(1/prim.N)# Initialize stationary distribution with uniform
        res = new(v, similar(v), 0, similar(v, Int), statdist,0)

        res
    end
end

## Solve Discrete Dynamic Program

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
function SolveProgram(prim::Primitives, v::Array{Float64,2};
  max_iter_vfi::Integer=500, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
    res = Results(prim, v)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi)
    create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
    res
end

#= Internal Utilities =#

## Bellman Operator

function bellman_operator!(prim::Primitives, v::Array{Float64,2})
  # initialize
  Tv = fill(-Inf,(prim.a_size,prim.s_size))
  sigma = zeros(prim.a_size,prim.s_size)

  # find max value for each (a,s) combination
  for state_index in 1:prim.s_size
    s = prim.s_vals[state_index]

    #= exploit monotonicity of policy function and only look for
    asset choices above the choice for previous asset level =#

    # Initialize lower bound of asset choices
    choice_lower = 1

      for asset_index in 1:prim.a_size
        a = prim.a_vals[asset_index]

        max_value = -Inf # initialize value for (a,s) combinations

          for choice_index in choice_lower:prim.a_size
            aprime = prim.a_vals[choice_index]
            c = s + a - prim.q*aprime
            if c > 0
              value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
              prim.beta*dot(prim.markov[state_index,:],v[choice_index,:])
              if value > max_value
                max_value = value
                sigma[asset_index,state_index] = choice_index
                choice_lower = choice_index
              end
            end
          end
        Tv[asset_index,state_index] = max_value
      end
  end
  Tv, sigma
end

## Value Function Iteration

function vfi!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)
    if prim.beta == 0.0
        tol = Inf
    else
        tol = epsilon
    end

    for i in 1:max_iter
        # updates Tv and sigma in place
        res.Tv, res.sigma = bellman_operator!(prim,res.v)

        # compute error and update the value with Tv inside results
        err = maxabs(res.Tv .- res.v)
        copy!(res.v, res.Tv)
        res.num_iter += 1

        if err < tol
            break
        end
    end

    res
end

## Find Stationary distribution

function create_statdist!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)

N = prim.N
m = prim.a_size

#= Create transition matrix Tstar that is N x N, where N is the number
of (a,s) combinations. Each row of Tstar is a conditional distribution over
(a',s') conditional on the (a,s) today defined by row index =#

Tstar = spzeros(N,N)

for state_today in 1:N
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
