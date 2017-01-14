using JuMP
using Mosek

m = Model(solver=MosekSolver())

@variable(m, -1.0 <= x <= 1.0, start = 0.9)
@variable(m, -1.0 <= y <= 1.0, start = 0.9)

testfun(x,y) = x^2 + y^2

JuMP.register(:testfun,2,testfun,autodiff=true)

@NLobjective(m, Min, testfun(x,y))

#@NLconstraint(m, abs(log(x))+abs(log(y)) <= 1.0)

solve(m)

obj = getobjectivevalue(m)
xval = getvalue(x)
yval = getvalue(y)
