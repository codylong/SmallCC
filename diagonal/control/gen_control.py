import cPickle as pickle
import subprocess
import os.path
import time

codystring  = 'codylong'
jimstring = 'jhhalverson'
outdir = 'output'

def writeScript(nmod,sigma,max_steps,eps,personstring,exp,mtype):
    out = "#!/bin/bash"
    out += "\n#SBATCH --job-name=control"+"k"+str(nmod)+'sig'+str(sigma)+"msteps"+str(max_steps) + 'exp' + exp + 'mtype' + mtype
    out += "\n#SBATCH --output=control"+"k"+str(nmod)+'sig'+str(sigma)+"msteps"+str(max_steps)+ 'exp' + exp + 'mtype' + mtype+".out"
    out += "\n#SBATCH --error=control"+"k"+str(nmod)+'sig'+str(sigma)+"msteps"+str(max_steps)+ 'exp' + exp + 'mtype' + mtype+".err"
    out += "\n#SBATCH --exclusive"
    out += "\n#SBATCH --partition=ser-par-10g-4"
    out += "\n#SBATCH -N 1"
    out += "\n#SBATCH --workdir=/home/" + personstring + "/wishart/control/workdir"
    out += "\ncd /home/" + personstring + "/wishart/control/"
    out += "\nmpirun -prot -srun -n 32 python ControlClasses.py " + str(max_steps) +" "+ exp+ " "+ str(nmod) + " " + str(eps) + " " + str(sigma) + ' ' + mtype
            
    f = open("/home/" + personstring + "/wishart/control/scripts/control"+"k"+str(nmod)+'sig'+str(sigma)+"msteps"+str(max_steps)+ 'exp' + exp + 'mtype' + mtype+".job",'w')
    f.write(out)
    f.close()
    output=subprocess.Popen("sbatch /home/" + personstring +  "/wishart/control/scripts/control"+"k"+str(nmod)+'sig'+str(sigma)+"msteps"+str(max_steps)+ 'exp' + exp + 'mtype' + mtype+".job",shell=True,stdout=subprocess.PIPE).communicate()[0]
    return output

for nmod in [10,25]:
    for exp in ['rand','grid']:
        writeScript(nmod,1e-4,2000000000,1e-50,codystring,exp,'load')

# June 1 evening runs, just changing eps relative to morning
# for eps in [1e-15]:
    # for sigma in [1e-4]:
    #     for nmod in [10,25]:
    #         for max_steps in ['10k', '100k']:
    #             for beta in [.01,.1,1,10]:
    #                 for power in [1]: # used to be 1 3 2
    #                     for gam in [.9,.95,.99,.999999]:
    #                         for arch in ["FFSoftmax","LSTMFR"]:
    #                             writeScript(eps,nmod,sigma,max_steps,jimstring,beta,power,gam,arch), 1000000000
                            
