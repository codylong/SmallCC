from gurobipy import *

# Create a new model
m = Model("qp")

nmod = 3
metrica = [[0 for ii in range(nmod)] for jj in range(nmod)]
for ii in range(nmod):
	metrica[ii][ii] = .01
metrica[0][1] = .0001
metrica[1][0] = .0001

metric1 = metrica
var = []
for j in range(len(metric1)):
    var.append(m.addVar(ub=1.0, vtype=GRB.INTEGER))

thr = 1
metric = metric1
obj = -1*thr
for ii in range(len(metric)):
	for jj in range(len(metric)):
		obj += metric[ii][jj]*var[ii]*var[jj]

m.setObjective(obj)



cons = 0
for ii in range(len(metric1)):
	for jj in range(len(metric1)):
		cons += metric1[ii][jj]*var[ii]*var[jj]

m.addConstr(cons - 1 >= 0, "c0")


m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())
