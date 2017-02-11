#=
Program Name: smm.jl
Estimate parameters from AR1 using Simulated Method of Moments
=#

using StatsBase
using PyPlot
using QuantEcon: meshgrid
using Optim: optimize

include("smm_utilities.jl")

# create data generating process
dgp = DGP()

# simulate series of true data from GDP
true_data = generate_true(dgp,T=200)

# simulate r.v. from N(0,1)
sim_rv = generate_rv(T=200,H=10)

# calculate data moments
MT = data_moments(true_data)

function smm(sim_rv::Array,MT::Tuple,n::Int;skipgraph="no")

  ## Estimate bhat_1 using W = I and optimize J

  if n == 2 || n == 0
    W = eye(2)
  elseif n == 3
    W = eye(3)
  elseif n == 1
    W = eye(1)
  else
    throw(ArgumentError("n must be 1, 2, 3, or 0"))
  end

  bhat_1 = optimize(b->J(W,b,MT,sim_rv,n=n),[0.5,1.0]).minimizer

  if skipgraph != "yes"

    # graph J in three dimensions over specified parameter range

    meshsize = 100

    rho_range = linspace(0.35,0.65,meshsize)
    sigma_range = linspace(0.8,1.2,meshsize)

    Jp = zeros(Float64,meshsize,meshsize)
    for i in 1:meshsize
      for j in 1:meshsize
        Jp[i,j] = J(W,[rho_range[i];sigma_range[j]],MT,sim_rv,n=n)
      end
    end

    savetitle = string("Jplot_",n)

    Jplotfig = figure()
    ax = Jplotfig[:gca](projection="3d")
    ax[:set_zlim](0.0,0.5)
    xgrid, ygrid = meshgrid(rho_range,sigma_range)
    title("J(b)")
    ax[:plot_surface](xgrid,ygrid,Jp,rstride=2,
      cstride=2,cmap=ColorMap("jet"),alpha=0.7,linewidth=0.25)

  end

  ## Estimate bhat_2 using Newey-West estimator

  S_hat = NeweyWest(sim_rv,bhat_1,n=n)
  Wstar_hat = inv(S_hat)

  bhat_2 = optimize(b->J(Wstar_hat,b,MT,sim_rv,n=n),[0.5,1.0]).minimizer

  ## Calculate standard errors

  Vhat, SE, gradient_gTH = standard_errors(sim_rv,bhat_2,Wstar_hat,n=n)

  ## Compute J statistic

  J_stat = J_test(sim_rv,bhat_2,Wstar_hat,MT,n=n)

  return bhat_1, bhat_2, Vhat, SE, gradient_gTH, J_stat
end

just_id1 = smm(sim_rv,MT,2)
just_id2 = smm(sim_rv,MT,0)
over_id = smm(sim_rv,MT,3)

#= Bootstrap finite sample distribution of the estimators =#

function bootstrap(;T::Int=100,H::Int=10,R::Int=100)
  # initialize vectors to keep track of estimators
  bhat1_vec = zeros(Float64,R,2)
  bhat2_vec = zeros(Float64,R,2)

  for i in 1:R
    # draw simulated data
    sim_rv = generate_rv(T=200,H=10)

    bhat1_vec[i,:], bhat2_vec[i,:] = smm(sim_rv,MT,3,skipgraph="yes")[1:2]
  end

  return bhat1_vec, bhat2_vec
end

bhat1_boot, bhat2_boot = bootstrap(R=1000)

## Plot histograms

# rho

rhohist = figure()
plt[:hist](bhat1_boot[:,1],50,color="red",edgecolor="none",alpha=0.2,label="bhat1")
plt[:hist](bhat2_boot[:,1],50,color="blue",alpha=0.2,label="bhat2")
plt[:legend]()
title("rho")

# sigma

sigmahist = figure()
plt[:hist](bhat1_boot[:,2],50,color="red",edgecolor="none",alpha=0.2,label="bhat1")
plt[:hist](bhat2_boot[:,2],50,color="blue",alpha=0.2,label="bhat2")
plt[:legend]()
title("sigma")
