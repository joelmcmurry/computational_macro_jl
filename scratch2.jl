using Optim

testfun(x) = x[1]^2 + x[2]^2

lower = [-1.0,-1.0]
upper = [1.0,1.0]
initial_pt = (lower+upper)/2
opt_results = optimize(DifferentiableFunction(testfun),
  initial_pt,lower,upper,Fminbox())

function f(x::Vector)
  return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]
results = optimize(DifferentiableFunction(f), initial_x, lower, upper, Fminbox(),  GradientDescent)


using BlackBoxOptim

testfun(x) = x[1]^2 + x[2]^2

bboptimize(testfun; SearchRange = (-1.0,1.0), NumDimensions=2)
