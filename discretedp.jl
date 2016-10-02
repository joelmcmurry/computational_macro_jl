#=
Program Name: Discrete Dynamic Program
Adapated from code by Daisuke Oyama, Spencer Lyon, and Matthew Mckay
available at QuantEcon.net
=#

using QuantEcon: s_wise_max

#= Type Discrete Dynamic Program. Inputs are current return matrix,
  transition matrix, and discount factor =#

type DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind}
  R::Array{T,NR} ## current return matrix
  Q::Array{T,NQ} ## transition matrix (3 dimensional)
  beta::Tbeta ## discount factor

    function DiscreteDP(R::Array, Q::Array, beta::Real)
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
        R_max = s_wise_max(R)
        if any(R_max .== -Inf)
            # First state index such that all actions yield -Inf
            state = find(R_max .== -Inf) #-Only Gives True
            throw(ArgumentError("for every state the reward must be
            finite for some action: violated for state $state"))
        end

        new(R, Q, beta)
    end
end

## Constructor for DiscreteDP object

DiscreteDP{T,NQ,NR,Tbeta}(R::Array{T,NR}, Q::Array{T,NQ}, beta::Tbeta) =
    DiscreteDP{T,NQ,NR,Tbeta,Int}(R, Q, beta)

## Bellman Operator

function bellman_operator!(ddp::DiscreteDP, v::Vector,
  Tv::Vector, sigma::Vector)
    vals = ddp.R + ddp.beta * ddp.Q * v
    find_policy(ddp, vals, Tv, sigma)
    Tv, sigma
end

## Find optimal policy

function find_policy!(vals::AbstractMatrix, out::Vector,
    out_argmax::Vector)
    # naive implementation where I just iterate over the rows
    nr, nc = size(vals)
    for i_r in 1:nr
        # reset temporaries
        cur_max = -Inf
        out_argmax[i_r] = 1

        for i_c in 1:nc
            @inbounds v_rc = vals[i_r, i_c]
            if v_rc > cur_max
                out[i_r] = v_rc
                out_argmax[i_r] = i_c
                cur_max = v_rc
            end
        end

    end
    out, out_argmax
end

# ## Bellman Operator
#
# function bellman_operator!(ddp::DiscreteDP, v::Vector,
#   Tv::Vector, sigma::Vector)
#     vals = ddp.R + ddp.beta * ddp.Q * v
#     find_policy(ddp, vals, Tv, sigma)
#     Tv, sigma
# end
#
# ## Value Function Iteration
#
# function vfi(ddp::DiscreteDP, ddpr::DPSolveResult{VFI},
#   max_iter::Integer, epsilon::Real, k::Integer)
#     if ddp.beta == 0.0
#         tol = Inf
#     else
#         tol = epsilon * (1-ddp.beta) / (2*ddp.beta)
#     end
#
#     for i in 1:max_iter
#         # updates Tv in place
#         bellman_operator(ddp, ddpr)
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
