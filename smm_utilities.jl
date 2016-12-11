#=
Program Name: smm_utilities.jl
Utilities for Simulated Method of Moments
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

# Simulate series of true data from GDP

function generate_true(dgp::DGP;T=200)
  true_data = zeros(Float64,T)
  true_data[1] = dgp.rho0*dgp.x0 + randn()*dgp.sigma0
  for t in 2:T
    true_data[t] = dgp.rho0*true_data[t-1] + randn()*dgp.sigma0
  end
 return true_data
end

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

## Calculate moments

function data_moments(true_data::Array{Float64})
  m1 = mean(true_data)
  m2 = var(true_data)
  m3 = autocov(true_data,[1])[1]
  return m1, m2, m3
end

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

## Objective function (up to three moments)

# n=0 option for using moments 2 and 3

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
  elseif n == 0
    J_b = g_TH[2:3]'*W[1:2,1:2]*g_TH[2:3]
  else
    throw(ArgumentError("J only defined for up to 3 moments"))
  end
  return J_b[1]
end

## Newey-West estimator for var-covar matrix S to estimate W*

function NeweyWest(sim_rv::Array,bhat::Array;iT::Int=4,n=2)
  T = length(sim_rv[:,1])
  H = length(sim_rv[1,:])
  # calcualte model moments with bhat
  MTH = model_moments(sim_rv,bhat[1],bhat[2])

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
          if t-1 == 0
            gamma_vec = [sim_data[t,h] - MTH[1],
              (sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t,h] - mean(sim_data[:,h]))*
                (0.0 - mean(sim_data[:,h])) - MTH[3]]
            gamma_vec_lag = [sim_data[t-j,h] - MTH[1],
              (sim_data[t-j,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t-j,h] - mean(sim_data[:,h]))*
                (0.0 - mean(sim_data[:,h])) - MTH[3]]
          elseif t-j-1 == 0
            gamma_vec = [sim_data[t,h] - MTH[1],
              (sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t,h] - mean(sim_data[:,h]))*
                (sim_data[t-1,h] - mean(sim_data[:,h])) - MTH[3]]
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
        elseif n == 0
          if t-1 == 0
            gamma_vec = [(sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t,h] - mean(sim_data[:,h]))*
                (0.0 - mean(sim_data[:,h])) - MTH[3]]
            gamma_vec_lag = [(sim_data[t-j,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t-j,h] - mean(sim_data[:,h]))*
                (0.0 - mean(sim_data[:,h])) - MTH[3]]
          elseif t-j-1 == 0
            gamma_vec = [(sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t,h] - mean(sim_data[:,h]))*
                (sim_data[t-1,h] - mean(sim_data[:,h])) - MTH[3]]
            gamma_vec_lag = [(sim_data[t-j,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t-j,h] - mean(sim_data[:,h]))*
                (0.0 - mean(sim_data[:,h])) - MTH[3]]
          else
            gamma_vec = [(sim_data[t,h] - mean(sim_data[:,h]))^2 - MTH[2],
              (sim_data[t,h] - mean(sim_data[:,h]))*
                (sim_data[t-1,h] - mean(sim_data[:,h])) - MTH[3]]
            gamma_vec_lag = [(sim_data[t-j,h] - mean(sim_data[:,h]))^2 - MTH[2],
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

## Calculate standard errors

function standard_errors(sim_rv::Array,bhat::Array,Wstar_hat::Array;n=2)
  # check dimensions are correct
  if length(Wstar_hat[:,1]) != n && n != 0
    throw(ArgumentError("number of moments much match weight matrix"))
  end

  if n == 0 && length(Wstar_hat[:,1]) != 2
    throw(ArgumentError("number of moments much match weight matrix"))
  end

  T = length(sim_rv[:,1])
  # calculate model moments
  MTH = model_moments(sim_rv,bhat[1],bhat[2])

  # calculate gradient of g_TH

  # fix small deviation for derivative approximation

  s = 1e-10

  # approximate derivative in rho-direction
  b_rho = bhat - [s,0]
  MTH_rho = model_moments(sim_rv,b_rho[1],b_rho[2])
  if n == 1
    moment_dev = [MTH[1]-MTH_rho[1]]
  elseif n == 2
    moment_dev = [MTH[1]-MTH_rho[1] MTH[2]-MTH_rho[2]]
  elseif n == 3
    moment_dev = [MTH[1]-MTH_rho[1] MTH[2]-MTH_rho[2] MTH[3]-MTH_rho[3]]
  elseif n == 0
    moment_dev = [MTH[2]-MTH_rho[2] MTH[3]-MTH_rho[3]]
  else
    throw(ArgumentError("only defined for up to 3 moments"))
  end

  partial_rho = -moment_dev/s

  # approximate derivative in sigma-direction
  b_sigma = bhat - [0,s]
  MTH_sigma = model_moments(sim_rv,b_sigma[1],b_sigma[2])
  if n == 1
    moment_dev = [MTH[1]-MTH_sigma[1]]
  elseif n == 2
    moment_dev = [MTH[1]-MTH_sigma[1] MTH[2]-MTH_sigma[2]]
  elseif n == 3
    moment_dev = [MTH[1]-MTH_sigma[1] MTH[2]-MTH_sigma[2] MTH[3]-MTH_sigma[3]]
  elseif n == 0
    moment_dev = [MTH[2]-MTH_sigma[2] MTH[3]-MTH_sigma[3]]
  else
    throw(ArgumentError("only defined for up to 3 moments"))
  end

  partial_sigma = -moment_dev/s

  gradient_gTH = [partial_rho' partial_sigma']

  # calculate variance-covariance matrix

  Vhat = (1/T)*inv(gradient_gTH'*Wstar_hat*gradient_gTH)
  SE = [sqrt(Vhat[1,1]) sqrt(Vhat[2,2])]

  return Vhat, SE, gradient_gTH
end

## Compute J statistic

function J_test(sim_rv::Array,bhat::Array,W::Array,MT::Tuple;n=2)
  T = length(sim_rv[:,1])
  H = length(sim_rv[1,:])

  J_b = J(W,bhat,MT,sim_rv,n=n)

  J_stat = T*(H/(1+H))*J_b

  return J_stat
end
