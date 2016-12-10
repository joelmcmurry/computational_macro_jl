#=
Program Name: smm.jl
Simulated Method of Moments Estimation Example
=#

using StatsBase

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
  true_data[1] = randn()*sqrt(dgp.sigma0)
  for t in 2:T
    true_data[t] = dgp.rho0*true_data[t-1] + randn()*sqrt(dgp.sigma0)
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
    sim_data[t,:] = rho*sim_data[t-1,:] + sqrt(sigma)*sim_rv[t,:]
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
  return m1, m2, m3
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
  return J_b
end
