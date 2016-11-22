#=
Program Name: krusell_smith_model.jl
Creates Krusell-Smith Model and Utilities
=#

using QuantEcon: gridmake

## Create composite type to hold model primitives

type Primitives

  beta :: Float64 ## discount rate
  alpha :: Float64 ## risk aversion
  delta :: Float64 ## depreciation rate
  z :: Array{Float64} ## aggregate shocks
  epsilon :: Array{Float64} ## idiosyncratic shocks
  u :: Array{Float64} ## unemployment rate
  ebar :: Float64 ## labor efficiency
  L :: Array{Float64} ## aggregate labor
  K_ss :: Float64 ## steady state capital without aggregate shocks

  k_min :: Float64 ## minimum capital value
  k_max :: Float64 ## maximum capital value
  k_size :: Int64 ## size of capital grid
  k_vals :: Vector{Float64} ## capital grid

  K_min :: Float64 ## minimum aggregate capital value
  K_max :: Float64 ## maximum aggregate capital value
  K_size :: Int64 ## size of aggregate capital grid
  K_vals :: Vector{Float64}

  z_size :: Int64 ## number of z shocks
  epsilon_size :: Int64 ## number of epsilon shocks
  u_size :: Int64 ## number of u values

  transmat :: Array{Float64} ## transition matrix
  transmat_stat :: Array{Float64} ## stationary distribution

  N :: Int64 ## number of simulated households
  T :: Int64 ## number of simulation periods

end

#= Outer Constructor for Primitives =#

function Primitives(;beta::Float64=0.99, alpha::Float64=0.36, delta::Float64=0.025,
 z::Array{Float64}=[1.01 0.99],epsilon::Array{Int64}=[0  1],
 u::Array{Float64}=[0.04 0.10], ebar::Float64=0.3271,
 k_min::Float64=0.00, k_max::Float64=15.0, k_size::Int64=100,
 N::Int64=5000,T::Int64=11000)

  # Grids

  k_vals = linspace(k_min, k_max, k_size)
  z_size = length(z)
  epsilon_size = length(epsilon)
  u_size = length(u)

  # Aggregate labor

  L = [1-u[1] 1-u[2]]

  # Initial guess for K

  K_ss = 5.7163

  # Create transition matrix

  transmat, transmat_stat = create_transmat!(u)

  primitives = Primitives(beta,alpha,delta,z,epsilon,
   u,ebar,L,K_ss,k_min,k_max,k_size,k_vals,z_size,epsilon_size,
   u_size,transmat, transmat_stat,N,T)

  return primitives

end

## Type Results which holds results of the problem

type Results
    v_g::Array{Float64}
    v_b::Array{Float64}
    Tv_g::Array{Float64}
    Tv_b::Array{Float64}
    num_iter::Int
    sigma_g::Array{Int}
    sigma_b::Array{Int}
    w::Float64 # wage rate
    r::Float64 # rental rate
    K::Float64 # aggregate capital
    a0::Float64 # log linear approximation coefficient
    b0::Float64 # log linear approximation coefficient
    a1::Float64 # log linear approximation coefficient
    b1::Float64 # log linear approximation coefficient

    #= Initialize K with average of the capital grid endpoints,
    initialize prices using average z and average L =#

    function Results(prim::Primitives;
      K=(prim.k_min+prim.k_max)/2,
      w=(1-prim.alpha)*mean(prim.z)*(K/mean(prim.L))^prim.alpha,
      r = prim.alpha*mean(prim.z)*(K/mean(prim.L))^(prim.alpha-1),
      a0=0.095,b0=0.085,a1=0.999,b1=0.999)

        # Initialize value with zeroes
        v_g = v_b = zeros(prim.k_size,prim.u_size)

        res = new(v_g,v_b,similar(v_g),similar(v_b),0,
          similar(v_g, Int),similar(v_b, Int),
          w,r,K,a0,b0,a1,b1)

        res
    end

    # Version with supplied value functions
    function Results(prim::Primitives,v_g::Array,v_b::Array;
      K=(prim.k_max+prim.k_min)/2,
      w=(1-prim.alpha)*mean(prim.z)*(K/mean(prim.L))^prim.alpha,
      r = prim.alpha*mean(prim.z)*(K/mean(prim.L))^(prim.alpha-1),
      a0=0.095,b0=0.085,a1=0.999,b1=0.999)

        res = new(v_g,v_b,similar(v_g),similar(v_b),0,
          similar(v_g, Int),similar(v_b, Int),
          w,r,K,a0,b0,a1,b1)

        res
    end
end

## Solve Discrete Dynamic Program

# Without initial value function (will be initialized at zeros)
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

function bellman_operator!(prim::Primitives, w, r,
  v_g::Array{Float64}, v_b::Array{Float64})
  # initialize
  Tv_g = fill(-Inf,(prim.k_size,prim.u_size))
  Tv_b = fill(-Inf,(prim.k_size,prim.u_size))
  sigma_g = zeros(prim.k_size,prim.u_size)
  sigma_b = zeros(prim.k_size,prim.u_size)

  # find max value for each (k,epsilon,z) combination
  for  in 1:prim.s_size
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

# function vfi!(prim::Primitives, res::Results,
#   max_iter::Integer, epsilon::Real)
#     if prim.beta == 0.0
#         tol = Inf
#     else
#         tol = epsilon
#     end
#
#     for i in 1:max_iter
#         # updates Tv and sigma in place
#         res.Tv, res.sigma = bellman_operator!(prim,res.v)
#
#         # compute error and update the value with Tv inside results
#         err = maxabs(res.Tv .- res.v)
#         copy!(res.v, res.Tv)
#         res.num_iter += 1
#
#         if err < tol
#             break
#         end
#     end
#
#     res
# end

## Create transmition matrix

function create_transmat!(u::Array{Float64})

  # parameters of transition matrix:

  duration_ug=1.5
  ug=u[1]
  duration_good=8.0
  ub=u[2]
  duration_bad=8.0
  duration_ub=2.5

  # transition probabilities

  pgg00 = (duration_ug-1)/duration_ug
  pbb00 = (duration_ub-1)/duration_ub
  pbg00 = 1.25*pbb00
  pgb00 = 0.75*pgg00
  pgg01 = (ug - ug*pgg00)/(1-ug)
  pbb01 = (ub - ub*pbb00)/(1-ub)
  pbg01 = (ub - ug*pbg00)/(1-ug)
  pgb01 = (ug - ub*pgb00)/(1-ub)
  pgg = (duration_good-1)/duration_good
  pgb = 1 - (duration_bad-1)/duration_bad
  pgg10 = 1 - (duration_ug-1)/duration_ug
  pbb10 = 1 - (duration_ub-1)/duration_ub
  pbg10 = 1 - 1.25*pbb00
  pgb10 = 1 - 0.75*pgg00
  pgg11 = 1 - (ug - ug*pgg00)/(1-ug)
  pbb11 = 1 - (ub - ub*pbb00)/(1-ub)
  pbg11 = 1 - (ub - ug*pbg00)/(1-ug)
  pgb11 = 1 - (ug - ub*pgb00)/(1-ub)
  pbg = 1 - (duration_good-1)/duration_good
  pbb = (duration_bad-1)/duration_bad

  # matrix

  trans_mat = zeros(Float64,4,4)

  trans_mat[1,1] = pgg*pgg11
  trans_mat[2,1] = pbg*pbg11
  trans_mat[3,1] = pgg*pgg01
  trans_mat[4,1] = pbg*pbg01
  trans_mat[1,2] = pgb*pgb11
  trans_mat[2,2] = pbb*pbb11
  trans_mat[3,2] = pgb*pgb01
  trans_mat[4,2] = pbb*pbb01
  trans_mat[1,3] = pgg*pgg10
  trans_mat[2,3] = pbg*pbg10
  trans_mat[3,3] = pgg*pgg00
  trans_mat[4,3] = pbg*pbg00
  trans_mat[1,4] = pgb*pgb10
  trans_mat[2,4] = pbb*pbb10
  trans_mat[3,4] = pgb*pgb00
  trans_mat[4,4] = pbb*pbb00

  trans_mat_tmp = trans_mat^10000

  trans_mat_stat = zeros(Float64,4)

  for i=1:4
      trans_mat_stat[i] = trans_mat_tmp[i,i]
  end

  trans_mat_out = trans_mat'

  return trans_mat_out, trans_mat_stat

end
