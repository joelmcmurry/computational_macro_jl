#=
Program Name: interpolation_model.jl
Holds model for calculation of optimal growth through interpolation
=#

using QuantEcon: gridmake
using Interpolations

## Type Primitives

type Primitives
  beta :: Float64 ## discount rate
  alpha :: Float64 ## risk aversion
  r :: Array{Float64} ## gross interest rate
  w :: Array{Float64} ## wage
  #K :: Float64 ## aggregate capital stock
  #L :: Float64 ## aggregate labor supply
  k_min :: Float64 ## minimum capital choice
  k_max :: Float64 ## maximum capital choice
  K_min :: Float64 ## minimum aggregate capital choice
  K_max :: Float64 ## maximum aggregate capital choice
  k_size :: Int64 ## size of k grid
  K_size :: Int64 ## size of K grid
  k_vals :: Array{Float64} ## k grid
  K_vals :: Array{Float64} ## K grid
  #k_K_vals :: Array{Float64} ## k,K Cartesian Mesh
  k_indices :: Array{Float64} ## indices of k grid
  K_indices :: Array{Float64} ## indices of K grid
  #k_K_indices :: Array{Float64} ## indices of k,K mesh
end

## Outer Constructor for Primitives

function Primitives(;beta=0.99,alpha=0.36,
  k_min=0,k_max=0.5,K_min=0.15,K_max=0.25,k_size=100,K_size=100)

  # Grids

  k_vals = linspace(k_min,k_max,k_size)
  K_vals =  linspace(K_min,K_max,K_size)
  #k_K_vals = gridmake(k_vals,K_vals)
  k_indices = gridmake(1:k_size)
  K_indices = gridmake(1:K_size)
  #k_K_indices = gridmake(1:k_size,1:K_size)

  r = (alpha)*(K_vals.^(alpha-1))
  w = (1-alpha)*(K_vals.^(alpha))

  # primitives = Primitives(beta,alpha,r,w,K,L,k_min,k_max,
  #   K_min,K_max,k_size,K_size,k_vals,K_vals,k_K_vals,k_indices,
  #   K_indices,k_K_indices)
  primitives = Primitives(beta,alpha,r,w,k_min,k_max,
    K_min,K_max,k_size,K_size,k_vals,K_vals,k_indices,
    K_indices)

  return primitives

end

## Type Results

type Results
    v::Array{Float64} # value function
    Tv::Array{Float64} # Bellman return
    num_iter::Int # iterations to converge
    sigma::Array{Int} # policy function

    function Results(prim::Primitives)
        v = zeros(prim.k_size,prim.K_size) # Initialize value with zeroes
        # Initialize stationary distribution with uniform over no-bankruptcy
        res = new(v,similar(v), 0, similar(v,Int))

        res
    end

    # Version w/ initial v
    function Results(prim::Primitives,v::Array{Float64,2})
        # Initialize stationary distribution with uniform over no-bankruptcy
        res = new(v,similar(v), 0, similar(v,Int))

        res
    end
end

#= Internal Utilites =#

function bellman_linear(prim::Primitives,v::Array{Float64})
  itp = interpolate(v,BSpline(Linear()),OnGrid())

  for k_index in 1:prim.k_indices
    for K_index in 1:prim.k_indices
      objective(kprime) = -log(prim.r*prim.k_vals[k_index]+prim.w-kprime)
    end
  end


  for (i, k) in enumerate(grid)
      objective(kprime) = - log(k^alpha + (1-delta)*k-kprime) - beta * Aw[kprime]
      res = optimize(objective, 0, k^alpha + (1-delta)*k)
      Tw[i] = - objective(res.minimum)
  end
  return Tw

  Tv = zeros(prim.k_vals,prim.K_vals)
end
