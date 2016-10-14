#=
Program Name: bankruptcy_model.jl
Creates model for separating and pooling bankruptcy equilibria
=#

using QuantEcon: gridmake

## Type Primitives

type Primitives
  beta :: Float64 ## discount rate
  alpha :: Float64 ## risk aversion
  r :: Float64 ## risk free interest rate
  q_pool :: Float64 ## discount bond price pooling
  q_menu :: Array{Float64,2} ## menu of discount bond prices for separating
  rho :: Float64 ## legal record keeping tech parameter
  a_min_pool :: Float64 ## pooling contract borrowing constraint
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
  zero_index :: Int64 ## index of asset level of 0
end

## Outer Constructor for Primitives

function Primitives(;beta::Float64=0.8, alpha::Float64=1.5,
  r::Float64=0.04, q_pool::Float64=0.9, rho::Float64=0.9,
  a_min_pool::Float64=-0.525, a_min::Float64=-2.0, a_max::Float64=5.0,
  a_size::Int64=100, markov=[0.75 (1-0.75);0.75 (1-0.75)],
  s_vals = [1, 0.05])

  # Grids

  a_vals = union(linspace(a_min, 0, a_size/2),linspace(0, a_max, a_size/2+1))
  zero_index = convert(Int64,a_size/2)
  s_size = length(markov[1:2,1])
  a_indices = gridmake(1:a_size)
  N = a_size*s_size
  a_s_vals = gridmake(a_vals,s_vals)
  a_s_indices = gridmake(1:a_size,1:s_size)
  q_menu = ones(a_size,2)*q_pool

  primitives = Primitives(beta, alpha, r, q_pool, q_menu, rho, a_min_pool,
  a_min, a_max, a_size, a_vals, a_indices, s_size, s_vals, markov, a_s_vals,
  a_s_indices, N, zero_index)

  return primitives

end

## Type Results

type Results
    v0::Array{Float64,2} # value function for no-bankrupt history
    v1::Array{Float64,2} # value function for bankrupt history
    Tv0::Array{Float64,2} # Bellman return for no-bankrupt history
    Tv1::Array{Float64,2} # Bellman return bankrupt history
    num_iter::Int
    sigma0::Array{Int,2} # asset policy function for no-bankrupt history
    sigma1::Array{Int,2} # asset policy function for bankrupt history
    d0::Array{Int,2} # default policy function
    statdist::Array{Float64}
    num_dist::Int

    function Results(prim::Primitives)
        v0 = zeros(prim.a_size,prim.s_size) # Initialise value with zeroes
        v1 = zeros(prim.a_size,prim.s_size)
        statdist = ones(prim.N)*(1/prim.N)# Initialize stationary distribution with uniform
        res = new(v0, v1, similar(v0), similar(v1), 0, similar(v0, Int),
        similar(v1,Int), similar(v0,Int), statdist, 0)

        res
    end

    # Version w/ initial v0, v1
    function Results(prim::Primitives,v0::Array{Float64,2},
      v1::Array{Float64,2})
        statdist = ones(prim.N)*(1/prim.N)# Initialize stationary distribution with uniform
        res = new(v0, v1, similar(v0), similar(v1), 0, similar(v0, Int),
        similar(v1,Int), similar(v0,Int), statdist, 0)

        res
    end
end

## Solve Discrete Dynamic Program

# Without initial value function (will be initialized at zeros)
function SolveProgram(prim::Primitives, bellman_clean::Function;
  max_iter_vfi::Integer=2000, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
    res = Results(prim)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi,bellman_clean)
    #create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
    res
end

# With Initial value functions v0, v1
function SolveProgram(prim::Primitives, v0::Array{Float64,2},
  v1::Array{Float64,2}, bellman_clean::Function;
  max_iter_vfi::Integer=2000, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-3)
    res = Results(prim, v0, v1)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi, bellman_clean)
    #create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
    res
end

#= Internal Utilities =#

## Bellman Operators

# Operator for defaulted agents (cannot borrow)

function bellman_defaulted!(prim::Primitives, v0::Array{Float64,2},
  v1::Array{Float64,2})
  # initialize output
  Tv1 = fill(-Inf,(prim.a_size,prim.s_size))
  sigma1 = zeros(Int64,prim.a_size,prim.s_size)

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
            # check if choice is greater than zero. (no borrowing)
            if prim.a_vals[choice_index] >= 0
              aprime = prim.a_vals[choice_index]
              c = s + a - (1/(1+prim.r))*aprime
              if c > 0
                value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
                prim.beta*dot(prim.markov[state_index,:],
                (prim.rho*v1[choice_index,:]+(1-prim.rho)*v0[choice_index,:]))
                if value > max_value
                  max_value = value
                  sigma1[asset_index,state_index] = choice_index
                  choice_lower = choice_index
                end
              end
            end
          end
        Tv1[asset_index,state_index] = max_value
      end
  end
  Tv1, sigma1
end

# Operators for agents that have not defaulted

# Pooling discount bond price

function bellman_clean_pool!(prim::Primitives, v0::Array{Float64,2},
    v1::Array{Float64,2})
  # initialize output
  Tv0 = fill(-Inf,(prim.a_size,prim.s_size))
  sigma0 = zeros(Int64,prim.a_size,prim.s_size)
  d0 = zeros(Int64,prim.a_size,prim.s_size)

  # initialize value functions for each choice (default, not default)
  Tvd = fill(-Inf,(prim.a_size,prim.s_size))
  Tvnd = fill(-Inf,(prim.a_size,prim.s_size))

  for state_index in 1:prim.s_size
  s = prim.s_vals[state_index]

  #= exploit monotonicity of policy function and only look for
  asset choices above the choice for previous asset level =#

  # Initialize lower bound of asset choices
  choice_lower = 1

    for asset_index in 1:prim.a_size
      a = prim.a_vals[asset_index]

      # if assets are negative, calculate value when defaulting
      if a < 0
        Tvd[asset_index, state_index] = (1/(1-prim.alpha))*
          (1/(s^(prim.alpha-1))-1) + prim.beta*
          dot(prim.markov[state_index,:],v1[prim.zero_index,:])
      end

      max_value = -Inf # initialize value for (a,s) combinations

        for choice_index in choice_lower:prim.a_size
          aprime = prim.a_vals[choice_index]
          c = s + a - prim.q_pool*aprime
          if c > 0
            value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
            prim.beta*dot(prim.markov[state_index,:],v0[choice_index,:])
            if value > max_value
              max_value = value
              sigma0[asset_index,state_index] = choice_index
              choice_lower = choice_index
            end
          end
        end
      Tvnd[asset_index,state_index] = max_value
      #= If defaulting value is higher, then value for that (a,s) is defaulting
      value and the decision rule is to default. Otherwise, value is
      not defaulting value and decision rule is not to default=#
      if Tvd[asset_index,state_index] > Tvnd[asset_index,state_index]
        d0[asset_index,state_index] = 1
        sigma0[asset_index,state_index] = 0
        Tv0[asset_index,state_index] = Tvd[asset_index,state_index]
      else
        Tv0[asset_index,state_index] = Tvnd[asset_index,state_index]
      end
    end
  end
  Tv0, sigma0, d0
end

# Separating discount bond price

function bellman_clean_sep!(prim::Primitives, v0::Array{Float64,2},
    v1::Array{Float64,2})
  # initialize output
  Tv0 = fill(-Inf,(prim.a_size,prim.s_size))
  sigma0 = zeros(Int64,prim.a_size,prim.s_size)
  d0 = zeros(Int64,prim.a_size,prim.s_size)

  # initialize value functions for each choice (default, not default)
  Tvd = fill(-Inf,(prim.a_size,prim.s_size))
  Tvnd = fill(-Inf,(prim.a_size,prim.s_size))

  for state_index in 1:prim.s_size
  s = prim.s_vals[state_index]

  #= exploit monotonicity of policy function and only look for
  asset choices above the choice for previous asset level =#

  # Initialize lower bound of asset choices
  choice_lower = 1

    for asset_index in 1:prim.a_size
      a = prim.a_vals[asset_index]

      # if assets are negative, calculate value when defaulting
      if a < 0
        Tvd[asset_index, state_index] = (1/(1-prim.alpha))*
          (1/(s^(prim.alpha-1))-1) + prim.beta*
          dot(prim.markov[state_index,:],v1[prim.zero_index,:])
      end

      max_value = -Inf # initialize value for (a,s) combinations

        for choice_index in choice_lower:prim.a_size
          aprime = prim.a_vals[choice_index]
          c = s + a - prim.q_menu[choice_index,state_index]*aprime
          if c > 0
            value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
            prim.beta*dot(prim.markov[state_index,:],v0[choice_index,:])
            if value > max_value
              max_value = value
              sigma0[asset_index,state_index] = choice_index
              choice_lower = choice_index
            end
          end
        end
      Tvnd[asset_index,state_index] = max_value
      #= If defaulting value is higher, then value for that (a,s) is defaulting
      value and the decision rule is to default. Otherwise, value is
      not defaulting value and decision rule is not to default=#
      if Tvd[asset_index,state_index] > Tvnd[asset_index,state_index]
        d0[asset_index,state_index] = 1
        sigma0[asset_index,state_index] = 0
        Tv0[asset_index,state_index] = Tvd[asset_index,state_index]
      else
        Tv0[asset_index,state_index] = Tvnd[asset_index,state_index]
      end
    end
  end
  Tv0, sigma0, d0
end

## Value Function Iteration

function vfi!(prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real, bellman_clean::Function)
    if prim.beta == 0.0
        tol = Inf
    else
        tol = epsilon
    end

    for i in 1:max_iter
        # Calculate

        # update Tv, sigma and replace in results
        res.Tv1, res.sigma1 = bellman_defaulted!(prim,res.v0,res.v1)
        res.Tv0, res.sigma0, res.d0 = bellman_clean(prim,res.v0,res.v1)

        # compute error and update the value with Tv inside results
        err0 = maxabs(res.Tv0 .- res.v0)
        err1 = maxabs(res.Tv1 .- res.v1)
        copy!(res.v0, res.Tv0)
        copy!(res.v1, res.Tv1)
        err = max(err0,err1)
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
