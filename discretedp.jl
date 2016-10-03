#=
Program Name: Discrete Dynamic Program
Adapated from code by Daisuke Oyama, Spencer Lyon, and Matthew Mckay
available at QuantEcon.net
=#

#= Type Discrete Dynamic Program. Inputs are current return matrix,
  transition matrix, and discount factor =#

type DiscreteProgram{T<:Real,NQ,NR,Tbeta<:Real}
  R::Array{T,NR} ## current return matrix
  Q::Array{T,NQ} ## transition matrix (3 dimensional)
  beta::Tbeta ## discount factor

    function DiscreteProgram(R::Array, Q::Array, beta::Real)
        # verify input integrity 1
        if NQ != 3
            msg = "Q must be 3-dimensional without state-action formulation"
            throw(ArgumentError(msg))
        end
        if NR != 2
            msg = "R must be 2-dimensional without state-action formulation"
            throw(ArgumentError(msg))
        end
        (beta < 0 || beta >= 1) &&  throw(ArgumentError("beta must be [0, 1)"))

        # verify input integrity 2
        num_states, num_actions = size(R)
        if size(Q) != (num_states, num_actions, num_states)
            throw(ArgumentError("shapes of R and Q must be (N,M) and (N,M,N)"))
        end

        # check feasibility
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

## Constructor for DiscreteProgram object

DiscreteProgram{T,NQ,NR,Tbeta}(R::Array{T,NR}, Q::Array{T,NQ}, beta::Tbeta) =
    DiscreteProgram{T,NQ,NR,Tbeta}(R, Q, beta)

## Type DPResult which holds results of the problems

type DPResult{Tval<:Real}
    v::Vector{Tval}
    Tv::Array{Tval}
    num_iter::Int
    sigma::Array{Int,1}
    #markov::MarkovChain

    function DPResult(ddp::DiscreteProgram)
        v = vec(maximum(ddp.R,2)) # Initialise value with current return max
        ddpr = new(v, similar(v), 0, similar(v, Int))

        # Fill in sigma with proper policy values
        (bellman_operator!(ddp, ddpr); ddpr.sigma)
        ddpr
    end

    # method to pass initial value function (skip the initialization)
    function DPResult(ddp::DiscreteProgram, v::Vector)
        ddpr = new(v, similar(v), 0, similar(v, Int))

        # Fill in sigma with proper policy values
        (bellman_operator!(ddp, ddpr); ddpr.sigma)
        ddpr
    end
end

## Bellman Operator

function bellman_operator!(ddp::DiscreteProgram, v::Vector,
  Tv::Vector, sigma::Vector)
    vals = ddp.R + ddp.beta * ddp.Q * v
    rowwise_max!(vals, Tv, sigma)
    Tv, sigma
end

#= Simplify input, telling the function to output Tv and sigma
to our results type=#

bellman_operator!(ddp::DiscreteProgram, ddpr::DPResult) =
  bellman_operator!(ddp, ddpr.v, ddpr.Tv, ddpr.sigma)

#= Find optimal policy. Iterate over each row (state) and column (action)
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

## Value Function Iteration
#
# function vfi(ddp::DiscreteProgram, ddpr::DPResult,
#   max_iter::Integer, epsilon::Real, k::Integer)
#     if ddp.beta == 0.0
#         tol = Inf
#     else
#         tol = epsilon * (1-ddp.beta) / (2*ddp.beta)
#     end
#
#     for i in 1:max_iter
#         # updates Tv in place
#         bellman_operator!(ddp, ddpr)
#
#         # compute error and update the v inside ddpr
#         err = maxabs(ddpr.Tv .- ddpr.v)
#         copy!(ddpr.v, ddpr.Tv)
#         ddpr.num_iter += 1
#
#         if err < tol
#             break
#         end
#     end
#
#     ddpr
# end
