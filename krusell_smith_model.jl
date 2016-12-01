#=
Program Name: krusell_smith_model.jl
Creates Krusell-Smith Model and Utilities
=#

using QuantEcon: gridmake
using Optim: optimize
using Interpolations

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

  r :: Array{Float64} ## rental rate
  w :: Array{Float64} ## wage rate

  k_min :: Float64 ## minimum capital value
  k_max :: Float64 ## maximum capital value
  k_size :: Int64 ## size of capital grid
  k_vals :: Vector{Float64} ## capital grid
  itp_k :: Interpolations.BSplineInterpolation ## interpolated capital grid

  K_min :: Float64 ## minimum aggregate capital value
  K_max :: Float64 ## maximum aggregate capital value
  K_size :: Int64 ## size of aggregate capital grid
  K_vals :: Vector{Float64} ## aggregate capital grid
  itp_K :: Interpolations.BSplineInterpolation ## interpolated aggregate capital grid

  z_size :: Int64 ## number of z shocks
  epsilon_size :: Int64 ## number of epsilon shocks
  u_size :: Int64 ## number of u values

  transmat :: Array{Float64} ## transition matrix
  transmat_stat :: Array{Float64} ## stationary distribution
  transmatagg :: Array{Float64}  ## transition matrix for aggregate shocks

  N :: Int64 ## number of simulated households
  T :: Int64 ## number of simulation periods

  ## capital motion approximation coefficients
  a0::Float64
  a1::Float64
  b0::Float64
  b1::Float64

end

#= Outer Constructor for Primitives =#

function Primitives(;beta::Float64=0.99, alpha::Float64=0.36, delta::Float64=0.025,
 z::Array{Float64}=[1.01 0.99],epsilon::Array{Int64}=[0  1],
 u::Array{Float64}=[0.04 0.10], ebar::Float64=0.3271,
 k_min::Float64=1.00e-2, k_max::Float64=15.0, k_size::Int64=100,
 K_min::Float64=1.00e-2, K_max::Float64=15.0, K_size::Int64=10,
 N::Int64=5000,T::Int64=11000,a0=0.095,b0=0.085,a1=0.999,b1=0.999)

  # Grids

  k_vals = linspace(k_min, k_max, k_size)
  K_vals = linspace(K_min, K_max, K_size)
  z_size = length(z)
  epsilon_size = length(epsilon)
  u_size = length(u)

  # Interpolated Grids

  itp_k = interpolate(k_vals,BSpline(Linear()),OnGrid())
  itp_K = interpolate(K_vals,BSpline(Linear()),OnGrid())

  # Aggregate (inelastic) labor supply

  L = [1-u[1] 1-u[2]]

  # prices

  w = [(1-alpha)*z[1]*(K_vals./L[1]).^alpha (1-alpha)*z[2]*(K_vals./L[2]).^alpha]
  r = [alpha*z[1]*(K_vals./L[1]).^(alpha-1) alpha*z[2]*(K_vals./L[2]).^(alpha-1)]

  # correct prices for K=0

  for K_index in 1:K_size
    if K_vals[K_index] == 0.0
      w[K_index,:] = [(1-alpha)*z[1]*(1e-10/L[1])^alpha (1-alpha)*z[2]*(1e-10/L[2])^alpha]
      r[K_index,:] = [alpha*z[1]*(1e-10/L[1])^(alpha-1) alpha*z[2]*(1e-10/L[2])^(alpha-1)]
    end
  end

  # Initial guess for K

  K_ss = 5.7163

  # Create transition matrix

  transmat, transmat_stat = create_transmat!(u)

  # Create transition matrix for aggregate shocks

  transmatagg = [1-(1/8) (1/8); (1/8) 1-(1/8)]

  primitives = Primitives(beta,alpha,delta,z,epsilon,
   u,ebar,L,K_ss,r,w,k_min,k_max,k_size,k_vals,itp_k,
   K_min,K_max,K_size,K_vals,itp_K,z_size,epsilon_size,
   u_size,transmat,transmat_stat,transmatagg,N,T,a0,a1,b0,b1)

  return primitives

end

## Type Results which holds results of the problem

type Results
    vg0::Array{Float64}
    vb0::Array{Float64}
    vg1::Array{Float64}
    vb1::Array{Float64}
    Tvg0::Array{Float64}
    Tvb0::Array{Float64}
    Tvg1::Array{Float64}
    Tvb1::Array{Float64}
    num_iter::Int
    sigmag0::Array
    sigmab0::Array
    sigmag1::Array
    sigmab1::Array

    function Results(prim::Primitives)

      # Initialize value with zeroes
      vg0 = vb0 = vg1 = vb1 = zeros(prim.k_size,prim.K_size)

      res = new(vg0,vb0,vg1,vb1,similar(vg0),similar(vb0),
        similar(vg1),similar(vb1),0,similar(vg0, Int),
        similar(vb0, Int),similar(vg1, Int),similar(vb1, Int))

      res
    end

    # Version with supplied value functions
    function Results(prim::Primitives,vg0::Array,vb0::Array,
      vg1::Array,vb1::Array)

      res = new(vg0,vb0,vg1,vb1,similar(vg0),similar(vb0),
        similar(vg1),similar(vb1),0,similar(vg0, Int),
        similar(vb0, Int),similar(vg1, Int),similar(vb1, Int))

      res
    end
end

## Generate decision rules

# Without initial value function (will be initialized at zeros)
function DecisionRules(T::Function, prim::Primitives;
  max_iter_vfi::Integer=1000, epsilon_vfi::Real=1e-2)
    res = Results(prim)
    vfi!(T, prim, res, max_iter_vfi, epsilon_vfi)
    res
end

# With Initial value functions
function DecisionRules(T::Function, prim::Primitives,
  vg1::Array, vb1::Array, vg0::Array, vb0::Array;
  max_iter_vfi::Integer=1000, epsilon_vfi::Real=1e-2)
    res = Results(prim, vg1, vb1, vg0, vb0)
    vfi!(T, prim, res, max_iter_vfi, epsilon_vfi)
    res
end

#= Internal Utilities =#

## Bellman Operator

# Grid search over k

function bellman_operator_grid!(prim::Primitives,
  vg1::Array{Float64},vb1::Array{Float64},
  vg0::Array{Float64},vb0::Array{Float64})
  # initialize
  Tvg1 = fill(-Inf,prim.k_size,prim.K_size)
  Tvb1 = fill(-Inf,prim.k_size,prim.K_size)
  Tvg0 = fill(-Inf,prim.k_size,prim.K_size)
  Tvb0 = fill(-Inf,prim.k_size,prim.K_size)
  sigmag1 = zeros(prim.k_size,prim.K_size)
  sigmab1 = zeros(prim.k_size,prim.K_size)
  sigmag0 = zeros(prim.k_size,prim.K_size)
  sigmab0 = zeros(prim.k_size,prim.K_size)

  # interpolate value functions
  itp_vg0 = interpolate(vg0,BSpline(Cubic(Line())),OnGrid())
  itp_vb0 = interpolate(vb0,BSpline(Cubic(Line())),OnGrid())
  itp_vg1 = interpolate(vg1,BSpline(Cubic(Line())),OnGrid())
  itp_vb1 = interpolate(vb1,BSpline(Cubic(Line())),OnGrid())

  # find max value for each (k,K,epsilon,z) combination
  for z_index in 1:prim.z_size
    for eps_index in 1:prim.epsilon_size
      # initialize floating Tv and sigma
      # Tv = ones(Float64,prim.k_size,prim.K_size)*(-2^51)
      Tv = fill(-Inf,prim.k_size,prim.K_size)
      sigma = zeros(prim.k_size,prim.K_size)

      # find row index of shocks for transition matrix
      if z_index == 1 && eps_index == 2
        shock_index = 1
      elseif z_index == 2 && eps_index == 2
        shock_index = 2
      elseif z_index == 1 && eps_index == 1
        shock_index = 3
      elseif z_index == 2 && eps_index == 1
        shock_index = 4
      end

      for K_index in 1:prim.K_size
        # approximate aggregate capital tomorrow (log linear form)
        if z_index == 1
          Kprime = exp(prim.a0 + prim.a1*log(prim.K_vals[K_index]))
        else
          Kprime = exp(prim.b0 + prim.b1*log(prim.K_vals[K_index]))
        end


        # find index of aggregate capital tomorrow
        findmatch(Kprime_index)=abs(prim.itp_K[Kprime_index]-Kprime)
        Kprime_index = optimize(Kprime_index->findmatch(Kprime_index),1.0,prim.K_size).minimum

        kprime_lower = 1 # initialize lower bound of asset choices
        for k_index in 1:prim.k_size

          max_value = -Inf # initialize value for combinations

            for kprime_index in kprime_lower:prim.k_size
              c = prim.r[K_index,z_index]*prim.k_vals[k_index] +
                prim.w[K_index,z_index]*prim.epsilon[eps_index] +
                (1-prim.delta)*prim.k_vals[k_index] - prim.k_vals[kprime_index]
              if c > 0
                value = log(c) +
                    prim.beta*
                    (prim.transmat[shock_index,1]*itp_vg1[kprime_index,Kprime_index]+
                    prim.transmat[shock_index,2]*itp_vb1[kprime_index,Kprime_index]+
                    prim.transmat[shock_index,3]*itp_vg0[kprime_index,Kprime_index]+
                    prim.transmat[shock_index,4]*itp_vb0[kprime_index,Kprime_index])
                if value > max_value
                  max_value = value
                  sigma[k_index,K_index] = kprime_index
                  kprime_lower = kprime_index
                end
              end
            end
          Tv[k_index,K_index] = max_value
        end
      end

      # fill in proper Tv and sigma
      if z_index == 1 && eps_index == 2
        Tvg1 = Tv
        sigmag1 = sigma
      elseif z_index == 2 && eps_index == 2
        Tvb1 = Tv
        sigmab1 = sigma
      elseif z_index == 1 && eps_index == 1
        Tvg0 = Tv
        sigmag0 = sigma
      elseif z_index == 2 && eps_index == 1
        Tvb0 = Tv
        sigmab0 = sigma
      end
    end
  end
  Tvg1, Tvb1, Tvg0, Tvb0, sigmag1, sigmab1, sigmag0, sigmab0
end

# Optimize over linear interpolated k

function bellman_operator_itp!(prim::Primitives,
  vg1::Array{Float64},vb1::Array{Float64},
  vg0::Array{Float64},vb0::Array{Float64})
  # initialize
  Tvg1 = fill(-Inf,prim.k_size,prim.K_size)
  Tvb1 = fill(-Inf,prim.k_size,prim.K_size)
  Tvg0 = fill(-Inf,prim.k_size,prim.K_size)
  Tvb0 = fill(-Inf,prim.k_size,prim.K_size)
  sigmag1 = zeros(Float64,prim.k_size,prim.K_size)
  sigmab1 = zeros(Float64,prim.k_size,prim.K_size)
  sigmag0 = zeros(Float64,prim.k_size,prim.K_size)
  sigmab0 = zeros(Float64,prim.k_size,prim.K_size)

  # interpolate value functions
  itp_vg0 = interpolate(vg0,BSpline(Cubic(Line())),OnGrid())
  itp_vb0 = interpolate(vb0,BSpline(Cubic(Line())),OnGrid())
  itp_vg1 = interpolate(vg1,BSpline(Cubic(Line())),OnGrid())
  itp_vb1 = interpolate(vb1,BSpline(Cubic(Line())),OnGrid())

  # find max value for each (k,K,epsilon,z) combination
  for z_index in 1:prim.z_size
    for eps_index in 1:prim.epsilon_size
      # initialize floating Tv and sigma
      # Tv = ones(Float64,prim.k_size,prim.K_size)*(-2^51)
      Tv = fill(-Inf,prim.k_size,prim.K_size)
      sigma = zeros(Float64,prim.k_size,prim.K_size)

      # find row index of shocks for transition matrix
      if z_index == 1 && eps_index == 2
        shock_index = 1
      elseif z_index == 2 && eps_index == 2
        shock_index = 2
      elseif z_index == 1 && eps_index == 1
        shock_index = 3
      elseif z_index == 2 && eps_index == 1
        shock_index = 4
      end

      for K_index in 1:prim.K_size
        # approximate aggregate capital tomorrow (log linear form)
        if z_index == 1
          Kprime = exp(prim.a0 + prim.a1*log(prim.K_vals[K_index]))
        else
          Kprime = exp(prim.b0 + prim.b1*log(prim.K_vals[K_index]))
        end


        # find index of aggregate capital tomorrow
        findmatch(Kprime_index)=abs(prim.itp_K[Kprime_index]-Kprime)
        Kprime_index = optimize(Kprime_index->findmatch(Kprime_index),1.0,prim.K_size).minimum

        for k_index in 1:prim.k_size
          if prim.r[K_index]*prim.k_vals[k_index] +
            prim.w[K_index]*prim.epsilon[eps_index] +
            (1-prim.delta)*prim.k_vals[k_index] > 0

            objective(kprime_index) = -log(prim.r[K_index]*prim.k_vals[k_index] +
              prim.w[K_index]*prim.epsilon[eps_index] + (1-prim.delta)*prim.k_vals[k_index]
              - prim.itp_k[kprime_index]) -
              prim.beta*
              (prim.transmat[shock_index,1]*itp_vg1[kprime_index,Kprime_index]+
              prim.transmat[shock_index,2]*itp_vb1[kprime_index,Kprime_index]+
              prim.transmat[shock_index,3]*itp_vg0[kprime_index,Kprime_index]+
              prim.transmat[shock_index,4]*itp_vb0[kprime_index,Kprime_index])
            lower = 1.0
            upper = findlast(kprime_index->kprime_index<=prim.r[K_index]*prim.k_vals[k_index] +
              prim.w[K_index]*prim.epsilon[eps_index] + (1-prim.delta)*prim.k_vals[k_index]
              ,prim.itp_k)
            if lower < upper
              optimization = optimize(kprime_index->objective(kprime_index),lower,upper)
              optimizer = optimization.minimum
            else
              optimizer = lower
            end

            Tv[k_index,K_index] = -objective(optimizer)
            sigma[k_index,K_index] = prim.itp_k[optimizer]
          end
        end
      end

      # fill in proper Tv and sigma
      if z_index == 1 && eps_index == 2
        Tvg1 = Tv
        sigmag1 = sigma
      elseif z_index == 2 && eps_index == 2
        Tvb1 = Tv
        sigmab1 = sigma
      elseif z_index == 1 && eps_index == 1
        Tvg0 = Tv
        sigmag0 = sigma
      elseif z_index == 2 && eps_index == 1
        Tvb0 = Tv
        sigmab0 = sigma
      end
    end
  end
  Tvg1, Tvb1, Tvg0, Tvb0, sigmag1, sigmab1, sigmag0, sigmab0
end

## Value Function Iteration

function vfi!(T::Function, prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)
    if prim.beta == 0.0
        tol = Inf
    else
        tol = epsilon
    end

    for i in 1:max_iter
        # updates Tv and sigma in place
        res.Tvg1, res.Tvb1, res.Tvg0, res.Tvb0,
          res.sigmag1, res.sigmab1, res.sigmag0, res.sigmab0 =
          T(prim,res.vg1,res.vb1,res.vg0,res.vb0)

        # compute error and update the value with Tv inside results
        errg1 = maxabs(res.Tvg1 .- res.vg1)
        errg0 = maxabs(res.Tvg0 .- res.vg0)
        errb1 = maxabs(res.Tvb1 .- res.vb1)
        errb0 = maxabs(res.Tvb0 .- res.vb0)
        err = max(errg1,errg0,errb1,errb0)

        println("Iter: ", res.num_iter, " error: ", err)

        res.vg1 = res.Tvg1
        res.vg0 = res.Tvg0
        res.vb1 = res.Tvb1
        res.vb0 = res.Tvb0
        res.num_iter += 1

        if err < tol
            break
        end
    end

    res
end

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
