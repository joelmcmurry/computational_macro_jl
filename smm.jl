#=
Program Name: smm.jl
Estimate parameters from AR1 using Simulated Method of Moments
=#

using StatsBase
using PyPlot
using QuantEcon: meshgrid
using Optim: optimize

## Create type for data-generating process

type DGP
  rho0::Float64
  sigma0::Float64
  x0::Float64
  b0::Array{Float64}
end

function DGP(;rho0::Float64=0.5,sigma0::Float64=1.0,x0::Float64=0.0)
  b0 = [rho0, sigma0]
  return DGP(rho0,sigma0,x0,b0)
end
dgp = DGP()

# Simulate series of true data from GDP

function generate_true(dgp::DGP;T=200)
  true_data = zeros(Float64,T)
  true_data[1] = randn()*dgp.sigma0
  for t in 2:T
    true_data[t] = dgp.rho0*true_data[t-1] + randn()*dgp.sigma0
  end
 return true_data
end
true_data = generate_true(dgp)

# Simulate r.v. from N(0,1)

function generate_rv(;H=10,T=200)
  sim_rv = zeros(Float64,T,H)
  for h in 1:H
    for t in 1:T
      sim_rv[t,h] = randn()
    end
  end
 return sim_rv
end
sim_rv = generate_rv()

## Calculate moments

function data_moments(true_data::Array{Float64})
  m1 = mean(true_data)
  m2 = var(true_data)
  m3 = autocov(true_data,[1])[1]
  return m1, m2, m3
end
MT = data_moments(true_data)

function model_moments(sim_rv::Array{Float64},rho::Float64,sigma::Float64)
  T = length(sim_rv[:,1])
  H = length(sim_rv[1,:])
  # generate simulated data using supplied r.v. and parameters
  sim_data = similar(sim_rv)
  sim_data[1,:] = sim_rv[1,:]
  for t in 2:T
    sim_data[t,:] = rho*sim_data[t-1,:] + sigma*sim_rv[t,:]
  end

  # calculate moments for each simulated vector of data
  m1_vec = zeros(Float64,H)
  m2_vec = zeros(Float64,H)
  m3_vec = zeros(Float64,H)
  for h in 1:H
    m1_vec[h] = mean(sim_data[:,h])
    m2_vec[h] = var(sim_data[:,h])
    m3_vec[h] = autocov(sim_data[:,h],[1])[1]
  end

  # take average across H simulated vector
  m1 = mean(m1_vec)
  m2 = mean(m2_vec)
  m3 = mean(m3_vec)
  return m1, m2, m3, sim_data
end

## Calculate objective function (up to three moments)

function J(W::Array,b::Array,MT::Tuple,sim_rv::Array;n::Int64=2)
  # calculate model moments given parameters
  MTH = model_moments(sim_rv,b[1],b[2])

  g_TH = [MT[1]-MTH[1], MT[2]-MTH[2], MT[3]-MTH[3]]

  if n == 1
    J_b = g_TH[1:1]'*W[1:1,1:1]*g_TH[1:1]
  elseif n == 2
    J_b = g_TH[1:2]'*W[1:2,1:2]*g_TH[1:2]
  elseif n == 3
    J_b = g_TH[1:3]'*W[1:3,1:3]*g_TH[1:3]
  else
    throw(ArgumentError("J only defined for up to 3 moments"))
  end
  return J_b[1]
end

#= Estimate just identified case =#

## Estimate bhat_1 using W = I and optimize J

W = eye(2)
bhat_1 = optimize(b->J(W,b,MT,sim_rv),[0.5,1.0]).minimizer

# graph J in three dimensions over specified parameter range

meshsize = 100

rho_range = linspace(0.35,0.65,meshsize)
sigma_range = linspace(0.8,1.2,meshsize)

Jp = zeros(Float64,meshsize,meshsize)
for i in 1:meshsize
  for j in 1:meshsize
    Jp[i,j] = J(W,[rho_range[i];sigma_range[j]],MT,sim_rv)
  end
end

Jplotfig = figure()
ax = Jplotfig[:gca](projection="3d")
ax[:set_zlim](0.0,0.5)
xgrid, ygrid = meshgrid(rho_range,sigma_range)
title("J(b)")
ax[:plot_surface](xgrid,ygrid,Jp,rstride=2,
  cstride=2,cmap=ColorMap("jet"),alpha=0.7,linewidth=0.25)
savefig("C:/Users/j0el/Documents/Wisconsin/899/Problem Sets/PS10/Pictures/Jplot.pgf")

## Estimate bhat_2 using Newey-West estimator

# calculate Newey-West estimator for var-covar matrix S to estimate W*

function NeweyWest(sim_rv::Array,bhat_1::Array;iT::Int=4,n=2)
  T = length(sim_rv[:,1])
  H = length(sim_rv[1,:])
  # calcualte model moments with bhat_1
  MTH = model_moments(sim_rv,bhat_1[1],bhat_1[2])

  sim_data = MTH[4]

  S_hat_y = 0.0
  for j in 0:iT
    # calculate Gamma_{j,TH}
    Gamma_jTH = 0.0
    for h in 1:H
      for t in j+1:T
        if n == 1
          gamma_vec = [sim_data[t,h] - MTH[1]]
          gamma_vec_lag = [sim_data[t-j,h] - MTH[1]]
        elseif n == 2
          gamma_vec = [sim_data[t,h] - MTH[1],
            (sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2]]
          gamma_vec_lag = [sim_data[t-j,h] - MTH[1],
            (sim_data[t-j,h] - mean(sim_data[:,h]))^2 - MTH[2]]
        elseif n == 3
          if t-1 == 0 # use 0 as lag value for first observation
            gamma_vec = [sim_data[t,h] - MTH[1],
              (sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t,h] - mean(sim_data[:,h]))*
                (0.0 - mean(sim_data[:,h])) - MTH[3]]
          elseif t-j-1 == 0
            gamma_vec_lag = [sim_data[t-j,h] - MTH[1],
              (sim_data[t-j,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t-j,h] - mean(sim_data[:,h]))*
                (0.0 - mean(sim_data[:,h])) - MTH[3]]
          else
            gamma_vec = [sim_data[t,h] - MTH[1],
              (sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t,h] - mean(sim_data[:,h]))*
                (sim_data[t-1,h] - mean(sim_data[:,h])) - MTH[3]]
            gamma_vec_lag = [sim_data[t-j,h] - MTH[1],
              (sim_data[t-j,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t-j,h] - mean(sim_data[:,h]))*
                (sim_data[t-j-1,h] - mean(sim_data[:,h])) - MTH[3]]
          end
        else
          throw(ArgumentError("only defined for up to 3 moments"))
        end

        Gamma_jTH += (1/(T*H))*gamma_vec*gamma_vec_lag'
      end
    end
    if j == 0
      S_hat_y += Gamma_jTH
    else
      S_hat_y += (1-(j/(iT+1)))*(Gamma_jTH + Gamma_jTH')
    end
  end

  S_hat = (1+(1/H))*S_hat_y
  return S_hat
end
S_hat = NeweyWest(sim_rv,bhat_1)

Wstar_hat = inv(S_hat)

# calculate bhat_2

bhat_2 = optimize(b->J(Wstar_hat,b,MT,sim_rv),[0.5,1.0]).minimizer
