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
  z_vals:: Array{Float64} ## values of idiosyncratic shock values
  z_markov :: Array{Float64} ## Markov process for idiosyncratic shock values
  z_ergodic :: Array{Float64} ## ergodic distribution for idiosyncratic shock values
  a_z_vals :: Array{Float64} ## array of states: (a,z) combinations
  a_z_indices :: Array{Int64} ## indices of states: (a,z) combinations
  M :: Int64 ## number of possible (a,z) combination
  ageeff :: Array{Float64} ## age efficiency profile
end

#= Outer Constructor for Primitives. Supplies default values, field
names, and creates grid objects=#

function Primitives(;N::Int64=66,JR::Int64=46,n::Float64=0.011,beta::Float64=0.97,
  gamma::Float64=0.42,sigma::Float64=2.0,delta::Float64=0.06,alpha::Float64=0.36,
  w::Float64=1.05,r::Float64=0.05,b::Float64=0.2,theta::Float64=0.11,
  a_min::Float64=0.0,a_max::Float64=5.0,a_size::Int64=100,z_vals=[3.0, 0.5],
  z_markov=[0.9261 (1-0.9261);(1-0.9811) 0.9811],z_ergodic=[0.2037 (1-0.2037)])

  # Grids

  a_vals = linspace(a_min, a_max, a_size)
  z_size = length(z_markov[:,1])
  a_indices = gridmake(1:a_size)
  M = a_size*z_size
  a_z_vals = gridmake(a_vals,z_vals)
  a_z_indices = gridmake(1:a_size,1:z_size)

  # Import age-efficiency profile file

  ageeff_file = open("ageeff.txt")
  ageeff = readdlm(ageeff_file)
  close(ageeff_file)

  primitives = Primitives(N, JR, n, beta, gamma, sigma, delta, alpha, w, r, b,
    theta, a_min, a_max, a_size, a_vals, a_indices, z_size, z_vals, z_markov,
    z_ergodic, a_z_vals, a_z_indices, M, ageeff)

  return primitives

end

## Type Results which holds results of the problem

type Results
    v_working_hi::Array{Float64}
    v_working_lo::Array{Float64}
    v_retired::Array{Float64}
    policy_working_hi::Array{Int64}
    policy_working_lo::Array{Int64}
    policy_retired::Array{Int64}
    labor_supply_hi::Array{Float64}
    labor_supply_lo::Array{Float64}
    ss_working_hi::Array{Float64}
    ss_working_lo::Array{Float64}
    ss_retired::Array{Float64}
    mu::Array{Float64}

    function Results(prim::Primitives)

      v_working_hi = fill(-Inf,prim.a_size,prim.JR-1)
      v_working_lo = fill(-Inf,prim.a_size,prim.JR-1)
      v_retired = fill(-Inf,prim.a_size,prim.N-prim.JR+1)

      policy_working_hi = zeros(Int64,prim.a_size,prim.JR-1)
      policy_working_lo = zeros(Int64,prim.a_size,prim.JR-1)
      policy_retired = zeros(Int64,prim.a_size,prim.N-prim.JR+1)

      labor_supply_hi = zeros(Float64,prim.a_size,prim.JR-1)
      labor_supply_lo = zeros(Float64,prim.a_size,prim.JR-1)

      ss_working_hi = zeros(Float64,prim.a_size,prim.JR-1)
      ss_working_lo = zeros(Float64,prim.a_size,prim.JR-1)
      ss_retired = zeros(Float64,prim.a_size,prim.N-prim.JR+1)

      mu = ones(Float64,prim.N)

      res = new(v_working_hi, v_working_lo, v_retired,
        policy_working_hi, policy_working_lo, policy_retired,
        labor_supply_hi, labor_supply_lo,
        ss_working_hi, ss_working_lo, ss_retired, mu)

        res
    end

end

#= Solve model and return user-requested value and policies for
one working year and one retired year. Defaults are first year
of working life and first year of retired life =#

function SolveProgram(prim::Primitives;
  return_working_age=1, return_retired_age=46)
    res = Results(prim)
    back_induction!(prim,res)
    create_steadystate!(prim,res)

    v_working = hcat(res.v_working_hi[:,return_working_age],
      res.v_working_lo[:,return_working_age])
    v_retired = res.v_retired[:,return_retired_age-prim.JR+1]
    policy_working = hcat(res.policy_working_hi[:,return_working_age],
      res.policy_working_lo[:,return_working_age])
    policy_retired = res.policy_retired[:,return_retired_age-prim.JR+1]
    labor_supply = hcat(res.labor_supply_hi[:,return_working_age],
      res.labor_supply_lo[:,return_working_age])

    return res, v_working, v_retired, policy_working, policy_retired, labor_supply
end

#= Internal Utilities =#

## Bellman Operators

# Operator for retired agent

function bellman_retired!(prim::Primitives, v::Array{Float64,1})
  # initialize
  Tv = fill(-Inf,prim.a_size)
  policy = zeros(Int64,prim.a_size)

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
        if c > 0.00
          value = (1/(1-prim.sigma))*(c^((1-prim.sigma)*prim.gamma)) +
          prim.beta*v[choice_index]
          if value > max_value
            max_value = value
            policy[asset_index] = choice_index
            choice_lower = choice_index
          end
        end
      end
    Tv[asset_index] = max_value
  end
  Tv, policy
end

# Operator for working agent

function bellman_working!(prim::Primitives, v::Array{Float64,2}, age::Int64)
  # initialize output
  Tv = fill(-Inf,(prim.a_size,prim.z_size))
  policy = zeros(Int64,prim.a_size,prim.z_size)
  labor = zeros(Float64,prim.a_size,prim.z_size)

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
        l = (prim.gamma*(1-prim.theta)*prim.ageeff[age]*z*prim.w -
          (1-prim.gamma)*((1+prim.r)*a-aprime))*
          (1/((1-prim.theta)*prim.ageeff[age]*z*prim.w))
        if l < 0.00
          l = 0.00
        elseif l > 1.00
          l = 1.00
        end
        c = prim.w*(1-prim.theta)*prim.ageeff[age]*l + (1+prim.r)*a - aprime
        if c > 0.00
          value = (1/(1-prim.sigma))*((c^prim.gamma*(1.00-l)^(1.00-prim.gamma))
          ^(1.00-prim.sigma)) + prim.beta*
          dot(prim.z_markov[z_index,:],v[choice_index,:])
          if value >= max_value
            max_value = value
            policy[asset_index,z_index] = choice_index
            labor[asset_index,z_index] = l
            choice_lower = choice_index
          end
        end
      end
    Tv[asset_index,z_index] = max_value
    end
  end
  Tv, policy, labor
end

## Backward Induction Procedures

function back_induction!(prim::Primitives,res::Results)
  # Initialize terminal period value
  vN = fill(-Inf,prim.a_size)

  # Calculate terminal period value
  for asset_index in 1:prim.a_size
    a = prim.a_vals[asset_index]
    c = (1+prim.r)*a + prim.b
    vN[asset_index] = (1/(1-prim.sigma))*(c^((1-prim.sigma)*prim.gamma))
  end

  # Backward induction to find value at beginning of retirement
  res.v_retired[:,prim.N-prim.JR+1] = vN
  res.policy_retired[:,prim.N-prim.JR+1] = ones(prim.a_size)
  vfloat_r = vN
  for i in 1:prim.N-prim.JR
    age = prim.N - i
    backward_index = age - prim.JR + 1
    age_optimization = bellman_retired!(prim,vfloat_r)
    vfloat_r = age_optimization[1]
    res.policy_retired[:,backward_index] = age_optimization[2]
    res.v_retired[:,backward_index] = vfloat_r
  end

  # Backward induction to find value at beginning of life
  vfloat_w = hcat(vfloat_r,vfloat_r)
  for i in 1:prim.JR-1
    age = prim.JR - i
    backward_index = age
    age_optimization = bellman_working!(prim,vfloat_w,age)
    vfloat_w = age_optimization[1]
    policy_working = age_optimization[2]
    labor_supply_working = age_optimization[3]
    res.v_working_hi[:,backward_index] = vfloat_w[:,1]
    res.v_working_lo[:,backward_index] = vfloat_w[:,2]
    res.policy_working_hi[:,backward_index] = policy_working[:,1]
    res.policy_working_lo[:,backward_index] = policy_working[:,2]
    res.labor_supply_hi[:,backward_index] = labor_supply_working[:,1]
    res.labor_supply_lo[:,backward_index] = labor_supply_working[:,2]
  end

  res
end

## Stationary distribution

function create_steadystate!(prim::Primitives,res::Results)

  # Find relative sizes of cohorts
  res.mu = ones(Float64,prim.N)
  for i in 1:prim.N-1
    res.mu[i+1]=res.mu[i]/(1+prim.n)
  end

  # Normalize so relative sizes sum to 1
  res.mu = res.mu/(sum(res.mu))

  # Reset steady state distributions

  res.ss_working_hi = zeros(Float64,prim.a_size,prim.JR-1)
  res.ss_working_lo = zeros(Float64,prim.a_size,prim.JR-1)
  res.ss_retired = zeros(Float64,prim.a_size,prim.N-prim.JR+1)

  #= Start newborn generation with 0 wealth and draw from ergodic
  distribution for ability =#

  for asset in 1:prim.a_size
    if prim.a_vals[asset] == 0.00
      res.ss_working_hi[asset,1] = res.mu[1]*prim.z_ergodic[1]
      res.ss_working_lo[asset,1] = res.mu[1]*prim.z_ergodic[2]
    else
      res.ss_working_hi[asset,1] = 0.00
      res.ss_working_lo[asset,1] = 0.00
    end
  end

  #= Use policy rules to calculate steady state asset holdings and ability
  at each age, weighted by relative cohort mass =#

  for age in 2:prim.N
    for asset in 1:prim.a_size # loop over asset holdings in previous year
      for choice_index in 1:prim.a_size # loop over possible policies
        if age < prim.JR # before retirement
          if res.policy_working_hi[asset,age-1] == choice_index
            res.ss_working_hi[choice_index,age] +=
              (res.mu[age]/res.mu[age-1])*res.ss_working_hi[asset,age-1]*prim.z_markov[1,1]
            res.ss_working_lo[choice_index,age] +=
              (res.mu[age]/res.mu[age-1])*res.ss_working_hi[asset,age-1]*prim.z_markov[1,2]
          end
          if res.policy_working_lo[asset,age-1] == choice_index
            res.ss_working_hi[choice_index,age] +=
              (res.mu[age]/res.mu[age-1])*res.ss_working_lo[asset,age-1]*prim.z_markov[2,1]
            res.ss_working_lo[choice_index,age] +=
              (res.mu[age]/res.mu[age-1])*res.ss_working_lo[asset,age-1]*prim.z_markov[2,2]
          end
        elseif age == prim.JR # at retirement
          if res.policy_working_hi[asset,age-1] == choice_index
            res.ss_retired[choice_index,1] +=
              (res.mu[age]/res.mu[age-1])*res.ss_working_hi[asset,age-1]
          end
          if res.policy_working_lo[asset,age-1] == choice_index
            res.ss_retired[choice_index,1] +=
              (res.mu[age]/res.mu[age-1])*res.ss_working_lo[asset,age-1]
          end
        else # after retirement
          if res.policy_retired[asset,age-prim.JR] == choice_index
            res.ss_retired[choice_index,age-prim.JR+1] +=
              (res.mu[age]/res.mu[age-1])*res.ss_retired[asset,age-prim.JR]
          end
        end
      end
    end
  end

  res
end
