#=
Program Name: krusell_smith_compute.jl
Runs Krusell-Smith model
=#

using PyPlot
using Interpolations

include("krusell_smith_model.jl")

# initialize model primitives

const prim = Primitives()

function k_s_compute(prim::Primitives,operator::Function)

  ## Start in the good state and simulate sequence of T aggregate shocks

  agg_shock_index = zeros(Int64,prim.T)
  agg_shock_vals = zeros(Float64,prim.T)

  agg_shock_index[1] = 1
  agg_shock_vals[1] = prim.z[1]
  for t in 2:prim.T
    if rand() <= prim.transmatagg[agg_shock_index[t-1],1]
      agg_shock_index[t] = 1
      agg_shock_vals[t] = prim.z[1]
    else
      agg_shock_index[t] = 2
      agg_shock_vals[t] = prim.z[2]
    end
  end

  # initialize matrix to store shocks

  idio_shock_vals = zeros(Float64,prim.N,prim.T)

  ## Start all agents employed and simulate sequence of T idiosyncratic shocks

  for i in 1:prim.N
    idio_shock_vals[i,1] = prim.epsilon[1]
  end

  #= calculate sequence of idiosyncratic shocks based on previous
  period aggreate and idiosyncratic shocks =#

  for t in 2:prim.T
    for i in 1:prim.N
      # calculate probability of being employed at time t
      if agg_shock_index[t-1] == 1 # last period good aggregate shock
        if idio_shock_vals[i,t-1] == 1.0 # last period employed
          prob_emp = prim.transmat[1,1]+prim.transmat[1,2]
        else # last period unemployed
          prob_emp = prim.transmat[3,1]+prim.transmat[3,2]
        end
      else  # last period bad aggregate shock
        if idio_shock_vals[i,t-1] == 1.0 # last period employed
          prob_emp = prim.transmat[2,1]+prim.transmat[2,2]
        else # last period unemployed
          prob_emp = prim.transmat[4,1]+prim.transmat[4,2]
        end
      end

      # draw shocks for time t
      if rand() <= prob_emp
        idio_shock_vals[i,t] = prim.epsilon[1]
      else
        idio_shock_vals[i,t] = prim.epsilon[2]
      end
    end
  end

  ## Start all agents with steady state capital holdings from complete mkt economy

  # initialize matrix to store capital holdings

  k_holdings = zeros(Float64,prim.N,prim.T)

  # initialize array to hold average capital holdings

  k_avg = zeros(Float64,prim.T)

  # endow each agent with K_ss

  for i in 1:prim.N
    k_holdings[i,1] = prim.K_ss
  end

  k_avg[1] = (1/prim.N)*sum(k_holdings[:,1])

  ## Calculate decision rules using chosen bellman Operator

  # store decision rules in results object

  res = DecisionRules(operator,prim)

  ## Using decision rules, populate matrices

  # interpolate policy functions

  itp_sigmag0 = interpolate(res.sigmag0,BSpline(Cubic(Line())),OnGrid())
  itp_sigmab0 = interpolate(res.sigmab0,BSpline(Cubic(Line())),OnGrid())
  itp_sigmag1 = interpolate(res.sigmag1,BSpline(Cubic(Line())),OnGrid())
  itp_sigmab1 = interpolate(res.sigmab1,BSpline(Cubic(Line())),OnGrid())

  findmatch(k_avg_index,target)=abs(prim.itp_K[k_avg_index]-target)
  for t in 1:prim.T-1
    # find index of avg capital time t
    targetK = k_avg[t]
    k_avg_index = optimize(k_avg_index->findmatch(k_avg_index,targetK),1.0,prim.K_size).minimum
    for i in 1:prim.N
      targetk = k_holdings[i,t]
      k_holdings_index = optimize(k_holdings_index->findmatch(k_holdings_index,targetk),1.0,prim.k_size).minimum
      if agg_shock_index[t] == 1 # good aggregate shock
        if idio_shock_vals[i,t] == 1.0 # employed
          k_holdings[i,t+1] = itp_sigmag1[k_holdings_index,k_avg_index]
        else # unemployed
          k_holdings[i,t+1] = itp_sigmag0[k_holdings_index,k_avg_index]
        end
      else  # bad aggregate shock
        if idio_shock_vals[i,t] == 1.0 # employed
          k_holdings[i,t+1] = itp_sigmab1[k_holdings_index,k_avg_index]
        else # unemployed
          k_holdings[i,t+1] = itp_sigmab0[k_holdings_index,k_avg_index]
        end
      end
    end
    k_avg[t+1] = (1/prim.N)*sum(k_holdings[:,t+1])
  end

  # drop first 1000 observations

  k_avg_trim = k_avg[1001:prim.T]
  agg_shock_index_trim = agg_shock_index[1001:prim.T]

  ## Regress log K' on log K to estimate forecasting coefficients

  # count number of good and bad periods

  g_period_count = 0
  b_period_count = 0
  for t in 1:length(agg_shock_index_trim)
    if agg_shock_index_trim[t] == 1
      g_period_count += 1
    else
      b_period_count += 1
    end
  end

  # split (avg k,avg k') into two datasets: good and bad periods

  k_avg_g = zeros(Float64,g_period_count,2)
  k_avg_b = zeros(Float64,b_period_count,2)

  # populate (avg k,avg k') in each datasets

  g_index = 1
  b_index = 1
  for t in 1:length(agg_shock_index_trim)-1
    if agg_shock_index_trim[t] == 1
      k_avg_g[g_index,1] = k_avg_trim[t]
      k_avg_g[g_index,2] = k_avg_trim[t+1]
      g_index += 1
    else
      k_avg_b[b_index,1] = k_avg_trim[t]
      k_avg_b[b_index,2] = k_avg_trim[t+1]
      b_index += 1
    end
  end



end
