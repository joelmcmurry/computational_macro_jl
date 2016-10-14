#=
Program Name: bankruptcy_model.jl
Creates model for separating and pooling bankruptcy equilibria
=#

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
end

## Outer Constructor for Primitives

function Primitives(;beta::Float64=0.8, alpha::Float64=1.5,
  r::Float64=0.04, q_pool::Float64=0.9, rho::Float64=0.9,
  a_min_pool::Float64=-0.525, a_min::Float64=-2.0, a_max::Float64=5.0,
  a_size::Int64=100, markov=[0.75 (1-0.75);0.75 (1-0.75)], s_vals = [1, 0.05])

  # Grids

  a_vals = linspace(a_min, a_max, a_size)
  s_size = length(markov[1:2,1])
  a_indices = gridmake(1:a_size)
  N = a_size*s_size
  a_s_vals = gridmake(a_vals,s_vals)
  a_s_indices = gridmake(1:a_size,1:s_size)
  q_menu = ones(a_size,2)*q_pool

  primitives = Primitives(beta, alpha, r, q_pool, q_menu, rho, a_min_pool,
  a_min, a_max, a_size, a_vals, a_indices, s_size, s_vals, markov, a_s_vals,
  a_s_indices, N)

  return primitives

end

## Type Results

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
