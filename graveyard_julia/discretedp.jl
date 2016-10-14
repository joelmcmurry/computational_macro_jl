#=
Program Name: Discrete Dynamic Program
Adapated from code by Daisuke Oyama, Spencer Lyon, and Matthew Mckay
available at QuantEcon.net
=#

#= Type Discrete Dynamic Program. Inputs are current return matrix,
  transition matrix, and discount factor. Object is this triple
  after error checking =#

#this be some crazy shit

type DiscreteProgram{T<:Real,NQ,NR,Tbeta<:Real}
  R::Array{T,NR} ## current return matrix
  Q::SparseMatrixCSC{T,NQ} ## transition matrix (3 dimensional)
  beta::Tbeta ## discount factor

    function DiscreteProgram(R::Array, Q::Array, beta::Real)

        # check feasibility of current returns
        R_max = vec(maximum(R,2))
        if any(R_max .== -Inf)
            # First state index such that all actions yield -Inf
            state = find(R_max .== -Inf) #-Only Gives True
            throw(ArgumentError("for every state the reward must be
            finite for some action: violated for state $state"))
        end

        new(R, Q, beta)
    end
end

#= Constructor for DiscreteProgram object. This is needed because the
inner constructor above is only defined for particular
DiscreteProgram{T,NQ,NR,Tbeta} and not the general DiscreteProgram.
Outer constructor defines method for general DiscreteProgram (which
only applies to inputs with the proper types). =#

DiscreteProgram{T,NQ,NR,Tbeta}(R::Array{T,NR}, Q::SparseMatrixCSC{T,NQ},
  beta::Tbeta) = DiscreteProgram{T,NQ,NR,Tbeta}(R, Q, beta)

## Type DPResult which holds results of the problem

type DPResult{Tval<:Real}
    v::Vector{Tval}
    Tv::Array{Tval}
    num_iter::Int
    sigma::Array{Int,1}
    statdist::Array{Tval}

    function DPResult(ddp::DiscreteProgram)
        v = vec(maximum(ddp.R,2)) # Initialise value with current return max
        ddpr = new(v, similar(v), 0, similar(v, Int), similar(v))

        # Fill in sigma with proper policy values
        #(bellman_operator!(ddp, ddpr); ddpr.sigma)

        ddpr
    end

end

## Solve Discrete Dynamic Program

function SolveProgram{T}(ddp::DiscreteProgram{T};
  max_iter_vfi::Integer=250, epsilon_vfi::Real=1e-3,
  max_iter_statdist::Integer=250, epsilon_statdist::Real=1e-5)
    ddpr = DPResult{T}(ddp)
    vfi!(ddp, ddpr, max_iter_vfi, epsilon_vfi)
    create_statdist!(ddp, ddpr, max_iter_statdist, epsilon_statdist)
    ddpr
end

##########= INTERNAL UTILITIES =##########

## Bellman Operator

#= v is an initial guess of value, Tv is updated value and is
overwritten, sigma is policy rule and is overwritten =#

function bellman_operator!(ddp::DiscreteProgram, v::Vector,
  Tv::Vector, sigma::Vector)
    vals = ddp.R + ddp.beta * (ddp.Q * v)
    rowwise_max!(vals, Tv, sigma)
    Tv, sigma
end

function bellman_operator!(ddp::DiscreteProgram, v::Vector,
  Tv::Vector, sigma::Vector)
  # initialize
  value = inf or something
  # find max
  for i in 1:states
      for j in 1:choices
        if R[i,j] > 0
          value = R[i,j] + beta*(markov )
          if value > max_value
            max_value = value
          end
        end

      end
  end
end

#= Simplify input, telling the function to output Tv and sigma
to our results structure ddpr =#

bellman_operator!(ddp::DiscreteProgram, ddpr::DPResult) =
  bellman_operator!(ddp, ddpr.v, ddpr.Tv, ddpr.sigma)

## Value Function Iteration

function vfi!(ddp::DiscreteProgram, ddpr::DPResult,
  max_iter::Integer, epsilon::Real)
    if ddp.beta == 0.0
        tol = Inf
    else
        tol = epsilon * (1-ddp.beta) / (2*ddp.beta)
    end

    for i in 1:max_iter
        # updates Tv and sigma in place
        bellman_operator!(ddp, ddpr)

        # compute error and update the value with Tv inside ddpr
        err = maxabs(ddpr.Tv .- ddpr.v)
        copy!(ddpr.v, ddpr.Tv)
        ddpr.num_iter += 1

        if err < tol
            break
        end
    end

    ddpr
end

## Find optimal policy

#=Iterate over each row (state) and column (action)
to find the highest valued column (action) for that row (state).
The output is a vector of maximized values for each row (state)
and their column (action) indices in the matrix
=#

function rowwise_max!(vals::AbstractMatrix, out::Vector,
    out_argmax::Vector)

    num_row, num_col = size(vals)
    for row_index in 1:num_row
        # reset temporaries
        cur_max = -Inf
        out_argmax[row_index] = 1

        for col_index in 1:num_col
            @inbounds val_rowcol = vals[row_index, col_index]
            if val_rowcol > cur_max
                out[row_index] = val_rowcol
                out_argmax[row_index] = col_index
                cur_max = val_rowcol
            end
        end

    end
    out, out_argmax
end

## Find Stationary distribution

function create_statdist!(ddp::DiscreteProgram, ddpr::DPResult,
  max_iter::Integer, epsilon::Real)

N = size(ddp.R)[1]
m = size(ddp.R)[2]

#= Create transition matrix Tstar that is N x N, where N is the number
of states. Each row of Tstar is a conditional distribution over
states tomorrow conditional on the state today defined by row index =#

Tstar = spzeros(N,N)

for state in 1:N
  for choice in 1:m
    if ddpr.sigma[state] == choice
      Tstar[state,:] = ddp.Q[state,choice,:]
    end
  end
end

#= Find stationary distribution. Start with a uniform distribution over
states and feed through Tstar matrix until convergence=#

# initialize with uniform distribution over states
statdist = ones(N)*(1/N)

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

  ddpr.statdist = statdist

  ddpr

end

## Q*v (taken directly from QuantEcon)

#= We want to be able to take the inner product of v and each
``third dimension" column in Q. Recall that each point in
two dimensions on Q represents a state and a choice. The
third dimension is a distribution over states tomorrow given
the state today and choice. Thus, the inner product of that
conditional distribution and v gives the expected value for
being in the conditioning state and choosing the conditioning
action =#

import Base.*

function *{T}(A::SparseMatrixCSC{T,3}, v::Vector)
  shape = size(A)
  size(v,1) == shape[end] || error("wrong dimensions")

  B = reshape(A, prod(shape[1:end-1]), shape[end])
  out = B*v

  return reshape(out, shape[1:end-1])
end
