from gurobipy import *

# Create a new model
m = Model("qp")

# Create variables
x = m.addVar(ub=10.0, name="x")
x.vType = GRB.INTEGER
y = m.addVar(ub=10.0, name="y")
y.vType = GRB.INTEGER

#let's take a simple metric of the form [[1,.01],[.01,1]
# Set objective:
obj = 1*x*x + 1*y*y + 2*.01*x*y - 1
m.setObjective(obj)

# Add quadratic constraint to enforce we get a positive solution
m.addConstr(1*x*x + 1*y*y + 2*.01*x*y -1 >= 0, "c0")

#find the solution
m.optimize()

#this throws the error: urobipy.GurobiError: Q matrix is not positive semi-definite (PSD)