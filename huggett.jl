#=
Program Name: huggett.jl
DESCRIBE THIS STUFF
Adapated from code by Victoria Gregory available at QuantEcon.net
=#

using QuantEcon: gridmake

## Create compostive type ``Huggett"

type Huggett
  beta :: Float64 ## discount rate
  alpha :: Float64 ## risk aversion
  q :: Float64 ## interest rate
  R :: Array{Float64} ## current return matrix
  Q :: Array{Float64} ## transition matrix (3 dimensional)
  a_min :: Float64 ## minimum asset value
  a_max :: Float64 ## maximum asset value
  a_size :: Int64 ## size of asset grid
  a_vals :: Vector{Float64} ## asset grid
  s_size :: Int64 ## number of earnings values
  markov :: Array{Float64,2} ## Markov process for earnings
  a_s_vals :: Array{Float64} ## array of states: (a,s) combinations
  a_s_indices :: Array{Int64} ## indices of states: (a,s) combinations
  N :: Int64 ## number of possible (a,s) combinations
end

#= Outer Constructor for Huggett. Supplies default values, field
names, and creates grid objects=#

function Huggett(;beta::Float64=0.9932, alpha::Float64=1.5,
  q::Float64=0.9, a_min::Float64=-2.0, a_max::Float64=5.0,
  a_size::Int64=100, markov=[0.97 (1-0.97);0.5 (1-0.5)],
  s_vals = [1, 0.5])

  # Grids

  a_vals = linspace(a_min, a_max, a_size)
  s_size = length(markov[1:2,1])
  N = a_size*s_size
  a_s_vals = gridmake(a_vals,s_vals)
  a_s_indices = gridmake(1:a_size,1:s_size)

  #= Transition matrix. This is a 3-dim object giving the probability of
  landing in state (a',s') given that you start in state (a,s). Precisely,
  this is a 3-dim matrix with height N (number of states), width a_size
  (number of asset choices), and length N. Thus, for each point in the
  2-dim space "length by width", the N-dim vector extending "up" from that
  point gives us the distribution over the N states (a',s') conditional on
  being in state (a,s) (length index) and choosing a' (width index)=#

  Q = spzeros(Float64, N, a_size, N)
  for stateprime_index in 1:N
      for choice_index in 1:a_size
          for state_index in 1:N
              s_index = a_s_indices[state_index, 2]
              sprime_index = a_s_indices[stateprime_index, 2]
              aprime_index = a_s_indices[stateprime_index, 1]
              if aprime_index == choice_index
                  Q[state_index, choice_index, stateprime_index] =
                  markov[s_index, sprime_index]
              end
          end
      end
  end

  ## Current return matrix

  ## initialize matrix
  R = fill(-Inf,N,a_size)

  ## create instance of Huggett with intialized matrix
  huggett = Huggett(beta, alpha, q, R, Q, a_min, a_max, a_size,
  a_vals, s_size, markov, a_s_vals, a_s_indices, N)

  #= construct current return matrix based on instance
  =#
  curr_return!(huggett, q)

  return huggett

end

function curr_return!(huggett::Huggett, q::Float64)

    # Set up R, current return matrix
    R = huggett.R
    alpha = huggett.alpha
    for choice_index in 1:huggett.a_size
        aprime = huggett.a_vals[choice_index]
        for state_index in 1:huggett.N
            a = huggett.a_s_vals[state_index, 1]
            s = huggett.a_s_vals[state_index, 2]
            c = s + a - q*aprime
            if c > 0
                R[state_index, choice_index] =
                (1/(1-alpha))*(1/(c^(alpha-1))-1)
            end
        end
    end

    # Replace initial current return matrix with constructed matrix
    huggett.q = q
    huggett.R = R
    huggett

end
