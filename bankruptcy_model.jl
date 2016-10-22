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
  a_min :: Float64 ## minimum asset value
  a_max :: Float64 ## maximum asset value
  a_min_sep :: Array{Float64,2} ## separating contract borrowing constraints
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
  a_min::Float64=-0.525, a_max::Float64=5.0, a_size::Int64=100,
  markov=[0.75 (1-0.75);(1-0.75) 0.75], s_vals = [1, 0.05])

  # Asset grize side must be even

  if iseven(a_size) != true
    msg = "Asset grid size must be an even integer"
    throw(ArgumentError(msg))
  end

  # Grids

  a_vals = sort(union(linspace(a_min,a_max,a_size-1),0))
  zero_index = find(a_vals.==0)[1]
  s_size = length(markov[1:2,1])
  a_indices = gridmake(1:a_size)
  N = a_size*s_size
  a_s_vals = gridmake(a_vals,s_vals)
  a_s_indices = gridmake(1:a_size,1:s_size)
  q_menu = ones(a_size,2)*q_pool
  a_min_sep = ones(a_size,2)*a_min

  primitives = Primitives(beta, alpha, r, q_pool, q_menu, rho, a_min,
  a_max, a_min_sep, a_size, a_vals, a_indices, s_size,
  s_vals, markov, a_s_vals, a_s_indices, N, zero_index)

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
    d0::Array{Int,2} # default policy function (discrete choice)
    statdist::Array{Float64} # stationary distribution
    num_dist::Int

    function Results(prim::Primitives)
        v0 = zeros(prim.a_size,prim.s_size) # Initialise value with zeroes
        v1 = zeros(prim.a_size,prim.s_size)
        # Initialize stationary distribution with uniform over no-bankruptcy
        statdist = vcat(ones(prim.N),zeros(prim.N))*(1/prim.N)
        res = new(v0, v1, similar(v0), similar(v1), 0, similar(v0, Int),
        similar(v1,Int), similar(v0,Int), statdist, 0)

        res
    end

    # Version w/ initial v0, v1
    function Results(prim::Primitives,v0::Array{Float64,2},
      v1::Array{Float64,2})
        # Initialize stationary distribution with uniform over no-bankruptcy
        statdist = vcat(ones(prim.N),zeros(prim.N))*(1/prim.N)
        res = new(v0, v1, similar(v0), similar(v1), 0, similar(v0, Int),
        similar(v1,Int), similar(v0,Int), statdist, 0)

        res
    end
end

## Solve Discrete Dynamic Program

# Without initial value function (will be initialized at zeros)
function SolveProgram(prim::Primitives, bellman_clean::Function;
  max_iter_vfi::Integer=2000, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-4)
    res = Results(prim)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi,bellman_clean)
    create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
    res
end

# With Initial value functions v0, v1
function SolveProgram(prim::Primitives, v0::Array{Float64,2},
  v1::Array{Float64,2}, bellman_clean::Function;
  max_iter_vfi::Integer=2000, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=500, epsilon_statdist::Real=1e-4)
    res = Results(prim, v0, v1)
    vfi!(prim, res, max_iter_vfi, epsilon_vfi, bellman_clean)
    create_statdist!(prim, res, max_iter_statdist, epsilon_statdist)
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

      # Bankrupt history values only defined for positive assets
      if a >= 0
        for choice_index in choice_lower:prim.a_size
          # check if choice is greater than zero. (no borrowing)
          if prim.a_vals[choice_index] >= 0
            aprime = prim.a_vals[choice_index]
            c = s + a - (1/(1+prim.r))*aprime
            if c > 0
              value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
              prim.beta*dot(prim.markov[state_index,:],
              (prim.rho*v1[choice_index,:]+
              (1-prim.rho)*v0[choice_index,:]))
              if value > max_value
                max_value = value
                sigma1[asset_index,state_index] = choice_index
                choice_lower = choice_index
              end
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

          # saving and borrowing at different rates
          if aprime <= 0 # borrow at market raets
            c = s + a - prim.q_menu[choice_index,state_index]*aprime
          else # save at risk-free rates
            c = s + a - (1/(1+prim.r))*aprime
          end

          if c > 0
            value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
            prim.beta*dot(prim.markov[state_index,:],v0[choice_index,:])
            if value > max_value
              max_value = value
              sigma0[asset_index,state_index] = choice_index
              choice_lower = choice_index
            end
          end
        end # asset choice
      Tvnd[asset_index,state_index] = max_value
      #= If defaulting value is higher, then value for that (a,s) is defaulting
      value and the decision rule is to default. Otherwise, value is
      not defaulting value and decision rule is not to default=#
      if Tvd[asset_index,state_index] > Tvnd[asset_index,state_index]
        d0[asset_index,state_index] = 1
        sigma0[asset_index,state_index] = prim.zero_index
        Tv0[asset_index,state_index] = Tvd[asset_index,state_index]
      else
        Tv0[asset_index,state_index] = Tvnd[asset_index,state_index]
      end
    end # asset
  end # earnings state
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

          # check if asset choice is above borrowing constraints
          if aprime >= prim.a_min_sep[choice_index,state_index]

            # saving and borrowing at different rates
            if aprime <= 0 # borrow at market raets
              c = s + a - prim.q_menu[choice_index,state_index]*aprime
            else # save at risk-free rates
              c = s + a - (1/(1+prim.r))*aprime
            end

            if c > 0
              value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
              prim.beta*dot(prim.markov[state_index,:],v0[choice_index,:])
              if value > max_value
                max_value = value
                sigma0[asset_index,state_index] = choice_index
                choice_lower = choice_index
              end
            end
          end # borrowing constraint
        end # asset choice
      Tvnd[asset_index,state_index] = max_value
      #= If defaulting value is higher, then value for that (a,s) is defaulting
      value and the decision rule is to default. Otherwise, value is
      not defaulting value and decision rule is not to default=#
      if Tvd[asset_index,state_index] > Tvnd[asset_index,state_index]
        d0[asset_index,state_index] = 1
        sigma0[asset_index,state_index] = prim.zero_index
        Tv0[asset_index,state_index] = Tvd[asset_index,state_index]
      else
        Tv0[asset_index,state_index] = Tvnd[asset_index,state_index]
      end
    end # asset
  end # earnings state
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
Nover2 = convert(Int64,N/2)

## Transition Matrix

Tstar = spzeros(2*N,2*N)

#= Here we distinguish between "state/history" which adds
bankruptcy history as a state variable and "state" which
is just (a,s) combination matching the primitive indexing =#

#= Distribution will be N states without a bankruptcy history
and N states with a bankruptcy history. Total 2N =#

for state_hist_today in 1:2*N

  # look in sigma0 if state/history has a no-bankrupt history
  if state_hist_today <= N
  lookup_state_today = state_hist_today # back out (a,s) combination index
  choice_index = res.sigma0[prim.a_s_indices[lookup_state_today,1],
  prim.a_s_indices[lookup_state_today,2]] # a' given (a,s)
  #= if agent doesn't default, restrict state/history tomorrow
  to no-bankrupt states/histories =#
    if res.d0[prim.a_s_indices[lookup_state_today,1],
      prim.a_s_indices[lookup_state_today,2]] != 1
      for state_hist_tomorrow in 1:N
        lookup_state_tomorrow = state_hist_tomorrow
        if prim.a_s_indices[lookup_state_tomorrow,1] == choice_index
          Tstar[state_hist_today,state_hist_tomorrow] =
            prim.markov[prim.a_s_indices[lookup_state_today,2]
            ,prim.a_s_indices[lookup_state_tomorrow,2]]
        end
      end
    else #= if agent defaults, mass distributed to assets=zero
      with a bankrupt history (split between employment states
      tomorrow) =#
      zero_tomorrow_high = N+prim.zero_index
      zero_tomorrow_low = N+Nover2+prim.zero_index
      Tstar[state_hist_today,zero_tomorrow_high] =
        prim.markov[prim.a_s_indices[lookup_state_today,2],1]
      Tstar[state_hist_today,zero_tomorrow_low] =
        prim.markov[prim.a_s_indices[lookup_state_today,2],2]
    end
  else # look in sigma1 if state/history has a bankrupt history
  lookup_state_today = state_hist_today-N # back out (a,s) combination index
  choice_index = res.sigma1[prim.a_s_indices[lookup_state_today,1],
  prim.a_s_indices[lookup_state_today,2]] # a' given (a,s)
  #= agents cannot default and cannot have a < 0 (flagged by policy
  of 0) =#
    if choice_index != 0
      #= agents end up in no-bankrupt history with probability (1-rho)=#
      for state_hist_tomorrow in 1:N
        lookup_state_tomorrow = state_hist_tomorrow
        if prim.a_s_indices[lookup_state_tomorrow,1] == choice_index
          Tstar[state_hist_today,state_hist_tomorrow] =
            (1-prim.rho)*prim.markov[prim.a_s_indices[lookup_state_today,2]
            ,prim.a_s_indices[lookup_state_tomorrow,2]]
        end
      end
      #= agents end up in bankrupt history with probability rho=#
      for state_hist_tomorrow in N+1:2*N
        lookup_state_tomorrow = state_hist_tomorrow-N
        if prim.a_s_indices[lookup_state_tomorrow,1] == choice_index
          Tstar[state_hist_today,state_hist_tomorrow] =
            prim.rho*prim.markov[prim.a_s_indices[lookup_state_today,2]
            ,prim.a_s_indices[lookup_state_tomorrow,2]]
        end
      end
    end
  end
end

#= Find stationary distributions. Start with any distribution over
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
