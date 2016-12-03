#=
Program Name: krusell_smith_compute.jl
Runs Krusell-Smith model
=#

using PyPlot
using Interpolations

include("krusell_smith_model.jl")

# initialize model primitives

const prim = Primitives(k_size=500)

function k_s_compute(prim::Primitives,operator::Function;
  max_iter=100,paramtol=5e-3)

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

  ## Calculate decision rules using chosen bellman operator

  res = DecisionRules(operator,prim)

  # Initialize Output Objects

  k_holdings_index = zeros(Float64,prim.N,prim.T)
  k_holdings_vals = zeros(Float64,prim.N,prim.T)
  k_avg = zeros(Float64,prim.T)
  r2_g = 0.0
  r2_b = 0.0

  for j in 1:max_iter
    tic()

    ## Start all agents with steady state capital holdings from complete mkt economy

    # initialize matrix to store capital holdings

    # policy indices
    k_holdings_index = zeros(Float64,prim.N,prim.T)

    # values
    k_holdings_vals = zeros(Float64,prim.N,prim.T)

    # initialize array to hold average capital holdings (values)

    k_avg = zeros(Float64,prim.T)

    # define K value-to-index function
    val_to_index_K(K_index,target)=abs(prim.itp_K[K_index]-target)

    # find index of steady state capital in complete markets economy (K_ss)
    target = prim.K_ss
    K_ss_index = optimize(K_index->val_to_index_K(K_index,target),1.0,prim.K_size).minimum

    # endow each agent with K_ss

    for i in 1:prim.N
      k_holdings_vals[i,1] = prim.K_ss
      k_holdings_index[i,1] = K_ss_index
    end

    k_avg[1] = (1/prim.N)*sum(k_holdings_vals[:,1])

    ## Using decision rules, populate matrices

    # interpolate policy functions

    # policy index

    itp_sigmag0 = interpolate(res.sigmag0,BSpline(Linear()),OnGrid())
    itp_sigmab0 = interpolate(res.sigmab0,BSpline(Linear()),OnGrid())
    itp_sigmag1 = interpolate(res.sigmag1,BSpline(Linear()),OnGrid())
    itp_sigmab1 = interpolate(res.sigmab1,BSpline(Linear()),OnGrid())

    # values

    itp_sigmag0vals = interpolate(res.sigmag0vals,BSpline(Linear()),OnGrid())
    itp_sigmab0vals = interpolate(res.sigmab0vals,BSpline(Linear()),OnGrid())
    itp_sigmag1vals = interpolate(res.sigmag1vals,BSpline(Linear()),OnGrid())
    itp_sigmab1vals = interpolate(res.sigmab1vals,BSpline(Linear()),OnGrid())

    #tic()
    for t in 1:prim.T-1
      #println("t: ", t)
      # find index of avg capital time t
      targetK = k_avg[t]
      k_avg_index = optimize(k_avg_index->val_to_index_K(k_avg_index,targetK),1.0,prim.K_size).minimum
      for i in 1:prim.N
        if agg_shock_index[t] == 1 # good aggregate shock
          if idio_shock_vals[i,t] == 1.0 # employed
            k_holdings_index[i,t+1] = itp_sigmag1[k_holdings_index[i,t],k_avg_index]
            k_holdings_vals[i,t+1] = itp_sigmag1vals[k_holdings_index[i,t],k_avg_index]
            #k_holdings_vals[i,t+1] = prim.itp_k[k_holdings_index[i,t+1]]
          else # unemployed
            k_holdings_index[i,t+1] = itp_sigmag0[k_holdings_index[i,t],k_avg_index]
            k_holdings_vals[i,t+1] = itp_sigmag0vals[k_holdings_index[i,t],k_avg_index]
            #k_holdings_vals[i,t+1] = prim.itp_k[k_holdings_index[i,t+1]]
          end
        else  # bad aggregate shock
          if idio_shock_vals[i,t] == 1.0 # employed
            k_holdings_index[i,t+1] = itp_sigmab1[k_holdings_index[i,t],k_avg_index]
            k_holdings_vals[i,t+1] = itp_sigmab1vals[k_holdings_index[i,t],k_avg_index]
            #k_holdings_vals[i,t+1] = prim.itp_k[k_holdings_index[i,t+1]]
          else # unemployed
            k_holdings_index[i,t+1] = itp_sigmab0[k_holdings_index[i,t],k_avg_index]
            k_holdings_vals[i,t+1] = itp_sigmab0vals[k_holdings_index[i,t],k_avg_index]
            #k_holdings_vals[i,t+1] = prim.itp_k[k_holdings_index[i,t+1]]
          end
        end
      end
      k_avg[t+1] = (1/prim.N)*sum(k_holdings_vals[:,t+1])
    end
    #toc()

    # drop first 1000 observations

    k_avg_trim = k_avg[1001:prim.T]
    agg_shock_index_trim = agg_shock_index[1001:prim.T]

    ## Regress log K' on log K to estimate forecasting coefficients

    # count number of good and bad periods (ignore last period)

    g_period_count = 0
    b_period_count = 0
    for t in 1:length(agg_shock_index_trim)-1
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

    # regress log(avg k) on log(avg k')

    data_g_log = log(k_avg_g)
    data_b_log = log(k_avg_b)

    OLS_g = linreg(data_g_log[:,1],data_g_log[:,2])
    OLS_b = linreg(data_b_log[:,1],data_b_log[:,2])

    # calculate R^2

    fitted_g = OLS_g[1] + OLS_g[2]*data_g_log[:,1]
    fitted_b = OLS_b[1] + OLS_b[2]*data_b_log[:,1]

    residual_g = data_g_log[:,2] - fitted_g
    residual_b = data_b_log[:,2] - fitted_b

    r2_g = 1 - sum(residual_g.^2)/sum((data_g_log[:,2]-mean(data_g_log[:,2])).^2)
    r2_b = 1 - sum(residual_g.^2)/sum((data_b_log[:,2]-mean(data_b_log[:,2])).^2)

    # check parameter distance and R2

    paramdist = max(maxabs([prim.a0;prim.a1]-[OLS_g[1];OLS_g[2]]),
      maxabs([prim.b0;prim.b1]-[OLS_b[1];OLS_b[2]]))

    println("Iter Completed: ",j," paramdist: ", paramdist,
      " r2_g: ", r2_g, " r2_b: ", r2_b)

    if paramdist < paramtol || r2_g >= 0.998 && r2_b >= 0.998
      toc()
      break
    end

    ## Update guess of prediction coefficients

    prim.a0, prim.a1 = OLS_g
    prim.b0, prim.b1 = OLS_b

    ## Recalculate decision rules with new prediction coefficients

    res = DecisionRules(operator,prim)
    toc()

  end
  prim, res, k_holdings_vals, k_holdings_index, k_avg, r2_g, r2_b
end

tic()
prim, res, k_holdings_vals, k_holdings_index, k_avg, r2_g, r2_b = k_s_compute(prim,bellman_operator_grid!)
toc()

# trim simulated series

k_avg_trim = k_avg[1001:prim.T]
k_holdings_vals_trim = k_holdings_vals[:,1001:prim.T]

## Plots

# Average K

Kfig = figure()
plot(linspace(1,prim.T-1000,prim.T-1000),k_avg_trim,color="blue",linewidth=2.0,label="Avg K")
xlabel("t")
ylabel("Avg K")
legend(loc="lower right")
title("Average Capital")
ax = PyPlot.gca()
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS8/Pictures/avg_K.pgf")

# Sample agent asset holdings

sample_size = 5
sample_agent_holding = zeros(Float64,sample_size,prim.T-1000)

for i in 1:sample_size
  agent_draw = rand(1:prim.N)
  sample_agent_holding[i,:] = k_holdings_vals_trim[agent_draw,:]
  sampletitle = string("Capital Holdings - Agent ",agent_draw)
  samplesave = string("samplek_",i)

  samplefig = figure()
  plot(linspace(1,prim.T-1000,prim.T-1000),sample_agent_holding[i,:],color="blue",linewidth=2.0)
  xlabel("t")
  ylabel("k holdings")
  legend(loc="lower right")
  title(sampletitle)
  ax = PyPlot.gca()
  savefig(string("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS8/Pictures/",samplesave,".pgf"))
end
