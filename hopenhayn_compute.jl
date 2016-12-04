#=
Program Name: hopenhayn_compute.jl
Runs Hopenhayn model
=#

using PyPlot

include("hopenhayn_model.jl")

# initialize model primitives

const prim = Primitives(n_size=10000,n_max=1500.0)

function hopenhayn_compute(prim::Primitives;
    max_iter_ec=1000,ec_tol=1e-3,max_iter_dist=1000,dist_tol=1e-3,
    max_iter_labor=50000,labor_tol=1e-2)

  # initialize results objects

  res = Results(prim)

  ## Find price that satisfied entry condition

  p_lower = 0.0 # lower bound for p
  p_upper = 100.0 # upper bound for p

  for i in 1:max_iter_ec

    # find decision rules and value functions

    res = DecisionRules(prim)

    # calculate entry condition EC(p)

    EC = dot(prim.nu,res.TWe)/prim.p - prim.ce

    # update price and stop when entry condition is close to satisfied

    if abs(EC) < ec_tol
      break
    end

    if EC > 0
      p_upper = prim.p # lower price
    else
      p_lower = prim.p# raise price
    end
    p_new = (p_upper + p_lower)/2

    prim.p = p_new

  end

  ## Find stationary distribution and entrant measure

  # guess M = 1

  res.M = 1.0

  labor_d = 0.00
  labor_s = 0.00

  for j in 1:max_iter_labor

    # find stationary distribution given guess for M

    for k in 1:max_iter_dist

      # find distribution next period

      muprime = zeros(Float64,prim.s_size)
      for sprime_index in 1:prim.s_size
        # incumbent firms
        for s_index in 1:prim.s_size
          muprime[sprime_index] +=
            (1-res.x[s_index])*prim.F[s_index,sprime_index]*res.mu[s_index]
        end
        # entrant firms
        for s_index in 1:prim.s_size
          muprime[sprime_index] +=
            (1-res.xe[s_index])*prim.F[s_index,sprime_index]*
            res.M*prim.nu[s_index]
        end
      end

      # update distribution and stop when stationarity obtained

      dist_err = maxabs(res.mu - muprime)

      if abs(dist_err) < dist_tol
        break
      end

      res.mu = muprime

    end

    # labor market clearing

    # labor demand

    labor_d = 0.00
    for s_index in 1:prim.s_size
      labor_d += res.nd[s_index]*res.mu[s_index]
      labor_d += res.M*res.nde[s_index]+prim.nu[s_index]
    end

    # labor supply

    profits = 0.00
    for s_index in 1:prim.s_size
      profits += (prim.p*prim.s_vals[s_index]*(res.nd[s_index])^prim.theta -
        res.nd[s_index] - prim.p*prim.cf)*res.mu[s_index]
      profits += res.M*(prim.p*prim.s_vals[s_index]*
      (res.nde[s_index])^prim.theta - res.nde[s_index] - prim.p*prim.ce)*
      prim.nu[s_index]
    end

    labor_s = 1/prim.A - profits

    # update M and stop when labor market clears

    LMC = maxabs(labor_d - labor_s)

    println("Iter: ", j, " M: ", res.M, " LMC: ", LMC)

    if abs(LMC) < labor_tol
      break
    end

    if LMC > 0
      res.M = 0.9999*res.M + 0.0001*0.0 # lower M
    else
      res.M = 0.9999*res.M + 0.0001*1.00 # raise M
    end

  end
  prim, res, labor_d, labor_s
end

tic()
prim, res, labor_d, labor_s = hopenhayn_compute(prim)
toc()
