import cPickle as pickle
import subprocess
import os.path
import time
from notebooks.sigthresh import *

codystring  = 'codylong'
jimstring = 'jhhalverson'

def writeScript(eps,nmod,sigma,max_steps,personstring):
    out = "#!/bin/bash"
    out += "\n#SBATCH --job-name=e"+str(eps)+"k"+str(nmod)+"s"+str(sigma)+"msteps"+max_steps
    out += "\n#SBATCH --output=e"+str(eps)+"k"+str(nmod)+"s"+str(sigma)+"msteps"+max_steps+".out"
    out += "\n#SBATCH --error=e"+str(eps)+"k"+str(nmod)+"s"+str(sigma)+"msteps"+max_steps+".err"
    out += "\n#SBATCH --exclusive"
    out += "\n#SBATCH --partition=ser-par-10g-4"
    out += "\n#SBATCH -N 1"
    out += "\n#SBATCH --workdir=/home/" + personstring + "/wishart/workdir"
    out += "\ncd /home/" + personstring + "/wishart/"
    out += "\nmpirun -prot -srun -n 1 python train_a3c_gym.py 32 --steps 20000000 --env cc-"+max_steps+"-v0 --outdir /home/" + personstring + "/wishart/output --gamma .999999 --eps " + str(eps) + " --nmod " + str(nmod) + " --sigma " + str(sigma)
            
    f = open("/home/" + personstring + "/wishart/scripts/e"+str(eps)+"k"+str(nmod)+"s"+str(sigma)+"msteps"+max_steps+".job",'w')
    f.write(out)
    f.close()
    output=subprocess.Popen("sbatch /home/" + personstring +  "/wishart/scripts/e"+str(eps)+"k"+str(nmod)+"s"+str(sigma)+"msteps"+max_steps+".job",shell=True,stdout=subprocess.PIPE).communicate()[0]
    return output

for eps in [1e-3,1e-5,1e-10,1e-15]:
    for nmod in [10,20,30]:
        #sigma = sigthresh(nmod,nmod,eps)/((100.0)**(1.0/(2.0*nmod)))
        sigma = (1e-5)*sigthresh(nmod,nmod,eps)
	for max_steps in ['eig-and-bump']:
            writeScript(eps,nmod,sigma,max_steps,codystring)
