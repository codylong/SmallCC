from gurobipy import *

# Create a new model
m = Model("qp")

metric1 = [[1.6792583553541959e-07, -5.976807293122852e-08, 6.604287944702793e-08, 6.096186407370309e-09, -2.3478934769838348e-08, 2.4319604639768427e-08, 1.2626338538286576e-08, 5.048052939012293e-08, -4.642395150192069e-09, -2.41174579138813e-08], [-5.976807293122852e-08, 7.35173696439e-08, -1.3649461227726454e-08, -2.079913862988188e-08, 5.253045965665985e-08, -1.014895183592806e-08, -1.8429742879520275e-08, -2.125627870719365e-08, -8.161830007477394e-09, -2.6592314593543016e-09], [6.604287944702793e-08, -1.3649461227726454e-08, 1.1630054979093366e-07, -6.006157170769788e-08, -9.687131096932852e-09, 6.657953853186457e-08, 1.5134370193710917e-08, -4.524314386228646e-08, -5.261835576412445e-08, -1.2788346200605383e-08], [6.096186407370309e-09, -2.079913862988188e-08, -6.006157170769788e-08, 1.1872158808338166e-07, -2.6219604852107305e-08, 9.55636950257234e-09, 2.535178256098267e-08, 4.066371752937604e-08, 6.221879553059309e-08, 1.1212694509236364e-08], [-2.3478934769838348e-08, 5.253045965665985e-08, -9.687131096932852e-09, -2.6219604852107305e-08, 1.603374067397813e-07, 3.880717460430586e-09, -4.0277698192969135e-08, 2.2033772646660105e-08, -2.148622298767471e-08, 3.9063594118924e-08], [2.4319604639768427e-08, -1.014895183592806e-08, 6.657953853186457e-08, 9.55636950257234e-09, 3.880717460430586e-09, 1.3714601440021534e-07, 2.0758061090703464e-08, -6.704349172531844e-08, -6.676068265724746e-08, 4.086927734700697e-08], [1.2626338538286576e-08, -1.8429742879520275e-08, 1.5134370193710917e-08, 2.535178256098267e-08, -4.0277698192969135e-08, 2.0758061090703464e-08, 3.457681560692106e-08, -7.284645066481048e-09, 3.2858284292233626e-08, -1.5431973456853983e-08], [5.048052939012293e-08, -2.125627870719365e-08, -4.524314386228646e-08, 4.066371752937604e-08, 2.2033772646660105e-08, -6.704349172531844e-08, -7.284645066481048e-09, 1.8251206390710622e-07, 3.2353242763084825e-08, 1.4009425115293886e-08], [-4.642395150192069e-09, -8.161830007477394e-09, -5.261835576412445e-08, 6.221879553059309e-08, -2.148622298767471e-08, -6.676068265724746e-08, 3.2858284292233626e-08, 3.2353242763084825e-08, 1.362292636039387e-07, -5.007344095626165e-08], [-2.41174579138813e-08, -2.6592314593543016e-09, -1.2788346200605383e-08, 1.1212694509236364e-08, 3.9063594118924e-08, 4.086927734700697e-08, -1.5431973456853983e-08, 1.4009425115293886e-08, -5.007344095626165e-08, 6.099121014871148e-08]]
nmod = 2
metrica = [[metric1[ii][jj] for ii in range(nmod)] for jj in range(nmod)]
#for ii in range(nmod):
#	metrica[ii][ii] = 1

metric1 = metrica
var = []
for j in range(len(metric1)):
    var.append(m.addVar(ub=1e5, vtype=GRB.INTEGER))

metric = metric1
obj = -1
for ii in range(len(metric)):
	for jj in range(len(metric)):
		obj += metric[ii][jj]*var[ii]*var[jj]

m.setObjective(obj)


cons = 0
for ii in range(len(metric1)):
	for jj in range(len(metric1)):
		cons += metric1[ii][jj]*var[ii]*var[jj]

m.addConstr(-cons <= -1, "c0")


m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())