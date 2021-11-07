from gurobipy import *

# Create a new model
m = Model("qp")

metric = [[1.6792583553541959e-07, -5.976807293122852e-08], [-5.976807293122852e-08, 7.35173696439e-08]]
nmod = 2
#for ii in range(nmod):
#	metrica[ii][ii] = 1

var = []
for j in range(nmod):
    var.append(m.addVar(ub=1e5, vtype=GRB.INTEGER))

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
