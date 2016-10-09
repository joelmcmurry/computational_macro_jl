#=
Program Name: huggett_compute.jl
=#

include("huggett.jl")
include("discretedp.jl")

## parameters

max_iter = 100
max_iter_vfi = 2000
epsilon = 1e-3

tic()

## Start with a guess for the discount bond price

q = 0.5

## Instantiate Huggett model

huggett = Huggett(q=q)

## Create Dynamic Program

huggettdp = DiscreteProgram(huggett.R,huggett.Q,huggett.beta)

for i in 1:max_iter

  # Solve dynamic program given q
  huggettres = SolveProgram(huggettdp,max_iter_vfi=max_iter_vfi)

  # Calculate net asset holdings using stationary distribution
  net_assets = dot(huggett.a_vals[huggettres.sigma],
    huggettres.statdist)

  # Adjust q (and stop if asset market clears)
  if abs(net_assets) < epsilon
      break
  elseif net_assets > 0
    qprime = q + (1-q)/2
  else
    qprime = q/2
  end
  q = qprime

  # Update current return matrix given new q
  curr_return!(huggett,q)

  # Replace current return matrix in dynamic program
  huggettdp.R = huggett.R

end
toc()
