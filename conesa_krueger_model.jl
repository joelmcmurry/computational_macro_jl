#=
Program Name: conesa_krueger_model.jl
Creates Conesa-Krueger OLG Model and Utilities
=#

using QuantEcon: gridmake

## Create compostive type to hold model primitives

type Primitives
  N :: Int64 ##  number of periods agent is alive
  JR :: Int64 ## retirement age
  n :: Float64 ## population growth rate
  beta :: Float64 ## discount rate
  gamma :: Float64 ## utility consumption weight (relative to leisure)
  sigma :: Float64 ## coefficient of rel. risk aversion
  delta :: Float64 ## capital depreciation rate
  alpha :: Float64 ## capital share
  w :: Float64 ## wage rate
  r :: Float64 ## interest rate
  b :: Float64 ## pension benefit
  theta :: Float64 ## social security tax
  a_min :: Float64 ## minimum asset value
  a_max :: Float64 ## maximum asset value
  a_size :: Int64 ## size of asset grid
  a_vals :: Vector{Float64} ## asset grid
  a_indices :: Array{Int64} ## indices of choices
  z_size :: Int64 ## number of idiosyncratic shock values
  z_vals:: Array{Float64,1} ## values of idiosyncratic shock values
  z_markov :: Array{Float64,2} ## Markov process for idiosyncratic shock values
  z_ergodic :: Array{Float64,2} ## ergodic distribution for idiosyncratic shock values
  a_z_vals :: Array{Float64} ## array of states: (a,z) combinations
  a_z_indices :: Array{Int64} ## indices of states: (a,z) combinations
  M :: Int64 ## number of possible (a,z) combination
end

#= Outer Constructor for Primitives. Supplies default values, field
names, and creates grid objects=#

function Primitives(;N::Int64=66,JR::Int64=46,n::Float64=0.011,beta::Float64=0.97,
  gamma::Float64=0.42,sigma::Float64=2,delta::Float64=0.06,alpha::Float64=0.36,
  w::Float64=1.05,r::Float64=0.05,b::Float64=0.2,theta::Float64=0.11,a_min::Float64=0.0,
  a_max::Float64=5.0,a_size::Int64=100,z_vals=[3.0;0.5],
  z_markov=[0.9261 (1-0.9261);(1-0.9811) 0.9811],z_ergodic=[0.2037 1-0.2037])

  # Grids

  a_vals = linspace(a_min, a_max, a_size)
  z_size = length(markov[:,1])
  a_indices = gridmake(1:a_size)
  M = a_size*z_size
  a_z_vals = gridmake(a_vals,z_vals)
  a_z_indices = gridmake(1:a_size,1:z_size)

  primitives = Primitives(N, JR, n, beta, gamma, sigma, delta, alpha, w, r, b,
    theta, a_min, a_max, a_size, a_vals, z_size, z_vals, z_markov, z_ergodic, a_z_vals,
    a_z_indices, M)

  return primitives

end

## Type Results which holds results of the problem HERE

## Solve Discrete Dynamic Program HERE

#= Internal Utilities =#

## Bellman Operators

# Operator for retired agent

function bellman_retired!(prim::Primitives, v::Array{Float64,1})
  # initialize
  Tv = fill(-Inf,prim.a_size)
  policy = zeros(prim.a_size)

  #= exploit monotonicity of policy function and only look for
  asset choices above the choice for previous asset level =#

  # Initialize lower bound of asset choices
  choice_lower = 1

  # find max value for each a
  for asset_index in 1:prim.a_size
    a = prim.a_vals[asset_index]

    max_value = -Inf # initialize value

      for choice_index in choice_lower:prim.a_size
        aprime = prim.a_vals[choice_index]
        c = (1+prim.r)*a + prim.b - aprime
        if c > 0
          value = (1/(1-prim.sigma))*(c^((1-prim.sigma)*prim.gamma)) +
          prim.beta*v[choice_index]
          if value > max_value
            max_value = value
            policy[asset_index,state_index] = choice_index
            choice_lower = choice_index
          end
        end
      end
    Tv[asset_index,state_index] = max_value
  end
  Tv, policy
end

# Operator for working agent

function bellman_working!(prim::Primitives, v::Array{Float64,2}, age::Int64)
  # initialize output
  Tv = fill(-Inf,(prim.a_size,prim.z_size))
  policy = zeros(Int64,prim.a_size,prim.z_size)

  # pull in age-efficiency value

  for z_index in 1:prim.z_size
  z = prim.z_vals[z_index]

  #= exploit monotonicity of policy function and only look for
  asset choices above the choice for previous asset level =#

  # Initialize lower bound of asset choices
  choice_lower = 1

    for asset_index in 1:prim.a_size
    a = prim.a_vals[asset_index]

    max_value = -Inf # initialize value for (a,z) combinations

      for choice_index in choice_lower:prim.a_size
        aprime = prim.a_vals[choice_index]
        # calculate optimal labor supply for choice of aprime
        l = (prim.gamma*(1-prim.theta)*z*prim.w - (1-prim.gamma)*((1+prim.r)*a-aprime))*
          (((1-prim.theta)*w*z)^(-1))

        c = s + a - (1/(1+prim.r))*aprime
        if c > 0.00
          value = (1/(1-prim.alpha))*(1/(c^(prim.alpha-1))-1) +
          prim.beta*dot(prim.markov[state_index,:],
          (prim.rho*v1[choice_index,:]+
          (1-prim.rho)*v0[choice_index,:]))
          if value >= max_value
            max_value = value
            sigma1[asset_index,state_index] = choice_index
            choice_lower = choice_index
          end
        end

      end
    Tv1[asset_index,state_index] = max_value
    end
  end
  Tv1, sigma1
end
