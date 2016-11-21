#=
Program Name: interpolation_model.jl
Holds model for calculation of optimal growth through interpolation
=#

using QuantEcon: gridmake
using Optim: optimize
using Interpolations

## Type Primitives

type Primitives
  beta :: Float64 ## discount rate
  alpha :: Float64 ## risk aversion
  r :: Array{Float64} ## gross interest rate
  w :: Array{Float64} ## wage
  k_min :: Float64 ## minimum capital choice
  k_max :: Float64 ## maximum capital choice
  K_min :: Float64 ## minimum aggregate capital choice
  K_max :: Float64 ## maximum aggregate capital choice
  k_size :: Int64 ## size of k grid
  K_size :: Int64 ## size of K grid
  k_vals :: Array{Float64} ## k grid
  K_vals :: Array{Float64} ## K grid
end

## Outer Constructor for Primitives

function Primitives(;beta=0.99,alpha=0.36,
  k_min=0,k_max=0.5,K_min=0.15,K_max=0.25,k_size=100,K_size=100)

  # Grids

  k_vals = linspace(k_min,k_max,k_size)
  K_vals =  linspace(K_min,K_max,K_size)
  k_indices = gridmake(1:k_size)
  K_indices = gridmake(1:K_size)

  r = (alpha)*(K_vals.^(alpha-1))
  w = (1-alpha)*(K_vals.^(alpha))

  primitives = Primitives(beta,alpha,r,w,k_min,k_max,
    K_min,K_max,k_size,K_size,k_vals,K_vals)

  return primitives

end

## Type Results

type Results
    v::Array{Float64} # value function
    Tv::Array{Float64} # Bellman return
    num_iter::Int # iterations to converge
    sigma::Array{Float64} # policy function

    function Results(prim::Primitives)
        v = zeros(prim.k_size,prim.K_size) # Initialize value with zeroes
        # Initialize stationary distribution with uniform over no-bankruptcy
        res = new(v,similar(v), 0, similar(v))

        res
    end

    # Version w/ initial v
    function Results(prim::Primitives,v::Array{Float64,2})
        # Initialize stationary distribution with uniform over no-bankruptcy
        res = new(v,similar(v), 0, similar(v))

        res
    end
end

## Solve Program

function SolveProgram(prim::Primitives,interpflag;
  max_iter_vfi::Integer=1000, epsilon_vfi::Real=1e-2)
    if interpflag == "Linear"
      res = Results(prim)
      vfi!(bellman_linear,prim, res, max_iter_vfi, epsilon_vfi)
      res
    elseif interpflag == "CubicLinear"
      res = Results(prim)
      vfi!(bellman_cubic_linear,prim, res, max_iter_vfi, epsilon_vfi)
      res
    else
      msg = "Interp. Argument must be 'Linear' or 'CubicLinear'"
      throw(ArgumentError(msg))
    end
end

#= Internal Utilites =#

## Bellman operators

# Linear interpolation

function bellman_linear(prim::Primitives,v::Array{Float64})
  itp_choice = interpolate(prim.k_vals,BSpline(Linear()),OnGrid())
  itp_value = interpolate(v,BSpline(Linear()),OnGrid())

  Tv = fill(-Inf,prim.k_size,prim.K_size)
  sigma = zeros(Float64,prim.k_size,prim.K_size)

  for K_index in 1:prim.K_size
    for k_index in 1:prim.k_size
      objective(kprime) = -log(prim.r[K_index]*prim.k_vals[k_index] +
        prim.w[K_index] - itp_choice[kprime]) -
        prim.beta*itp_value[kprime,K_index]
      lower = 1.0
      upper = findlast(kprime->kprime<prim.r[K_index]*prim.k_vals[k_index] +
        prim.w[K_index],itp_choice)
      opt_choice = optimize(kprime->objective(kprime),lower,upper)

      Tv[k_index,K_index] = -objective(opt_choice.minimum)
      sigma[k_index,K_index] = itp_choice[opt_choice.minimum]
    end
  end

  return Tv, sigma

end

# Cubic spline

function bellman_cubic(prim::Primitives,v::Array{Float64})
  itp_choice = interpolate(prim.k_vals,BSpline(Linear()),OnGrid())
  itp_value = interpolate(v,BSpline(Cubic(Line())),OnGrid())

  Tv = fill(-Inf,prim.k_size,prim.K_size)
  sigma = zeros(Float64,prim.k_size,prim.K_size)

  for K_index in 1:prim.K_size
    for k_index in 1:prim.k_size
      objective(kprime) = -log(prim.r[K_index]*prim.k_vals[k_index] +
        prim.w[K_index] - itp_choice[kprime]) -
        prim.beta*itp_value[kprime,K_index]
      lower = 1.0
      upper = findlast(kprime->kprime<prim.r[K_index]*prim.k_vals[k_index] +
        prim.w[K_index],itp_choice)
      opt_choice = optimize(kprime->objective(kprime),lower,upper)

      Tv[k_index,K_index] = -objective(opt_choice.minimum)
      sigma[k_index,K_index] = itp_choice[opt_choice.minimum]
    end
  end

  return Tv, sigma

end

# Cubic spline in k-dimension and linear in K-dimension

function bellman_cubic_linear(prim::Primitives,v::Array{Float64})
  itp_choice = interpolate(prim.k_vals,BSpline(Linear()),OnGrid())
  itp_value = interpolate(v,(BSpline(Cubic(Line())),BSpline(Linear())),OnGrid())

  Tv = fill(-Inf,prim.k_size,prim.K_size)
  sigma = zeros(Float64,prim.k_size,prim.K_size)

  for K_index in 1:prim.K_size
    for k_index in 1:prim.k_size
      objective(kprime) = -log(prim.r[K_index]*prim.k_vals[k_index] +
        prim.w[K_index] - itp_choice[kprime]) -
        prim.beta*itp_value[kprime,K_index]
      lower = 1.0
      upper = findlast(kprime->kprime<prim.r[K_index]*prim.k_vals[k_index] +
        prim.w[K_index],itp_choice)
      opt_choice = optimize(kprime->objective(kprime),lower,upper)

      Tv[k_index,K_index] = -objective(opt_choice.minimum)
      sigma[k_index,K_index] = itp_choice[opt_choice.minimum]
    end
  end

  return Tv, sigma

end

## Find fixed point

function vfi!(T::Function, prim::Primitives, res::Results,
  max_iter::Integer, epsilon::Real)
    if prim.beta == 0.0
        tol = Inf
    else
        tol = epsilon
    end

    for i in 1:max_iter
        # updates Tv and sigma in place
        res.Tv, res.sigma = T(prim,res.v)

        # compute error and update the value with Tv inside results
        err = maxabs(res.Tv .- res.v)
        copy!(res.v, res.Tv)
        res.num_iter += 1
        println("Iteration: ", res.num_iter, " Error: ", err)

        if err < tol
            break
        end
    end

    res
end
