#=
Program Name: Discrete Dynamic Program
Adapated from code by Daisuke Oyama, Spencer Lyon, and Matthew Mckay
available at QuantEcon.net
=#

type DiscreteDP{T<:Real,NQ,NR,Tbeta<:Real,Tind}
  R::Array{T,NR} ## current return matrix
  Q::Array{T,NQ} ## transition matrix (3 dimensional)
  beta::Tbeta ## discount factor
  a_indices::Nullable{Vector{Tind}} ## action indices
  a_indptr::Nullable{Vector{Tind}} ## action index pointers

    function DiscreteDP(R::Array, Q::Array, beta::Real)
        # verify input integrity 1
        if NQ != 3
            msg = "Q must be 3-dimensional"
            throw(ArgumentError(msg))
        end
        if NR != 2
            msg = "R must be 2-dimensional"
            throw(ArgumentError(msg))
        end
        (beta < 0 || beta >= 1) &&  throw(ArgumentError("beta must be [0, 1)"))

        # verify input integrity 2
        num_states, num_actions = size(R)
        if size(Q) != (num_states, num_actions, num_states)
            throw(ArgumentError("shapes of R and Q must be (N,M) and (N,M,N)"))
        end

        # check feasibility
        R_max = state_max_return(R)
        if any(R_max .== -Inf)
            # First state index such that all actions yield -Inf
            state = find(R_max .== -Inf) #-Only Gives True
            throw(ArgumentError("for every state the reward must be
            finite for some action: violated for state $state"))
        end

        # here the indices and indptr are null.
        _a_indices = Nullable{Vector{Int}}()
        a_indptr = Nullable{Vector{Int}}()

        new(R, Q, beta, _a_indices, a_indptr)
    end

    # Note: We left R, Q as type Array to produce more helpful error message with regards to shape.
    # R and Q are dense Arrays
    function DiscreteDP(R::AbstractArray, Q::AbstractArray, beta::Real,
                        s_indices::Vector, a_indices::Vector)
        # verify input integrity 1
        if NQ != 2
            throw(ArgumentError("Q must be 2-dimensional with s-a formulation"))
        end
        if NR != 1
            throw(ArgumentError("R must be 1-dimensional with s-a formulation"))
        end
        (beta < 0 || beta >= 1) && throw(ArgumentError("beta must be [0, 1)"))

        # verify input integrity (same length)
        num_sa_pairs, num_states = size(Q)
        if length(R) != num_sa_pairs
            throw(ArgumentError("shapes of R and Q must be (L,) and (L,n)"))
        end
        if length(s_indices) != num_sa_pairs
            msg = "length of s_indices must be equal to the number of s-a pairs"
            throw(ArgumentError(msg))
        end
        if length(a_indices) != num_sa_pairs
            msg = "length of a_indices must be equal to the number of s-a pairs"
            throw(ArgumentError(msg))
        end

        if _has_sorted_sa_indices(s_indices, a_indices)
            a_indptr = Array(Int64, num_states+1)
            _a_indices = copy(a_indices)
            _generate_a_indptr!(num_states, s_indices, a_indptr)
        else
            # transpose matrix to use Julia's CSC; now rows are actions and
            # columns are states (this is why it's called as_ptr not sa_ptr)
            m = maximum(a_indices)
            n = maximum(s_indices)
            msg = "Duplicate s-a pair found"
            as_ptr = sparse(a_indices, s_indices, 1:num_sa_pairs, m, n,
                            (x,y)->throw(ArgumentError(msg)))
            _a_indices = as_ptr.rowval
            a_indptr = as_ptr.colptr

            R = R[as_ptr.nzval]
            Q = Q[as_ptr.nzval, :]
        end

        # check feasibility
        aptr_diff = diff(a_indptr)
        if any(aptr_diff .== 0.0)
            # First state index such that no action is available
            s = find(aptr_diff .== 0.0)  # Only Gives True
            throw(ArgumentError("for every state at least one action
                must be available: violated for state $s"))
        end

        # package into nullables before constructing type
        _a_indices = Nullable{Vector{Tind}}(_a_indices)
        a_indptr = Nullable{Vector{Tind}}(a_indptr)

        new(R, full(Q), beta, _a_indices, a_indptr)
    end
end

#=
Internal Utilities
=#

## find maximum current return possible for each state
state_max_return(returnmat::AbstractMatrix)
  = vec(maximum(returnmat,2))



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

## Bellman Operator

function bellman_operator!(ddp::DiscreteDP, v::Vector,
  Tv::Vector, sigma::Vector)
    vals = ddp.R + ddp.beta * ddp.Q * v
    find_policy(ddp, vals, Tv, sigma)
    Tv, sigma
end

## Value Function Iteration

function vfi(ddp::DiscreteDP, ddpr::DPSolveResult{VFI},
  max_iter::Integer, epsilon::Real, k::Integer)
    if ddp.beta == 0.0
        tol = Inf
    else
        tol = epsilon * (1-ddp.beta) / (2*ddp.beta)
    end

    for i in 1:max_iter
        # updates Tv in place
        bellman_operator(ddp, ddpr)

        # compute error and update the v inside ddpr
        err = maxabs(ddpr.Tv .- ddpr.v)
        copy!(ddpr.v, ddpr.Tv)
        ddpr.num_iter += 1

        if err < tol
            break
        end
    end

    ddpr
end
