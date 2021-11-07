import cPickle as pickle
import subprocess
import os.path
import time
from notebooks.sigthresh import *
from continue_experiment import *

codystring  = 'codylong'
jimstring = 'jhhalverson'
outdir = 'output_cc'

def writeScript(eps,nmod,sigma,max_steps,personstring,beta,power,gamma,mtotal):
    out = "#!/bin/bash"
    out += "\n#SBATCH --job-name=e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps + 'gamma' + str(gamma)
    out += "\n#SBATCH --output=e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+ 'gamma' + str(gamma) + ".out"
    out += "\n#SBATCH --error=e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+'gamma' + str(gamma) + ".err"
    out += "\n#SBATCH --exclusive"
    out += "\n#SBATCH --partition=ser-par-10g-4"
    out += "\n#SBATCH -N 1"
    out += "\n#SBATCH --workdir=/home/" + personstring + "/wishart/workdir"
    out += "\ncd /home/" + personstring + "/wishart/"
    exp = WishartExperiments(sig = sigma, eps = eps,n = nmod, gam = gamma,bet =beta,max = max_steps, out_dir = "/home/" + personstring +  "/wishart/output_test/", pow = power,maxtotal = mtotal)
    run_string = exp.train()
    out += "\nmpirun -prot -srun -n 1 " + run_string
            
    f = open("/home/" + personstring + "/wishart/scripts/e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+".job",'w')
    f.write(out)
    f.close()
    output=subprocess.Popen("sbatch /home/" + personstring +  "/wishart/scripts/e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+".job",shell=True,stdout=subprocess.PIPE).communicate()[0]
    return output

for eps in [1e-15]:
    for sigma in [1e-4]:
        for nmod in [10]:
            for max_steps in ['quickrun']:
                for beta in [1]:
                    for power in [1]:
			for gamma in [.9]:
                        	writeScript(eps,nmod,sigma,max_steps,codystring,beta,power,gamma,1000)
