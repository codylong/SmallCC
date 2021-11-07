from gurobipy import *

# Create a new model
m = Model("qp")

nmod = 2
metric = [[0 for ii in range(nmod)] for jj in range(nmod)]
for ii in range(nmod):
	metric[ii][ii] = 1

var = []
for j in range(len(metric)):
    var.append(m.addVar(ub=1000, vtype=GRB.INTEGER))

obj = -1
for ii in range(len(metric)):
	for jj in range(len(metric)):
		obj += metric[ii][jj]*var[ii]*var[jj]

m.setObjective(obj)

cc = 1e-1

cons = 0
for ii in range(len(metric1)):
	for jj in range(len(metric1)):
		cons += metric1[ii][jj]*var[ii]*var[jj]

m.addConstr(cons >= 1 + cc, "c0")


m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())
