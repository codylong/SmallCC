import cPickle as pickle
import subprocess
import os.path
import time
from notebooks.sigthresh import *

codystring  = 'codylong'
jimstring = 'jhhalverson'
outdir = 'roundbumpgame'

def writeScript(eps,nmod,sigma,max_steps,personstring,beta,power,gam,arch):
    out = "#!/bin/bash"
    out += "\n#SBATCH --job-name=e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+"arch"+arch
    out += "\n#SBATCH --output=e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+"arch"+arch+".out"
    out += "\n#SBATCH --error=e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+"arch"+arch+".err"
    out += "\n#SBATCH --exclusive"
    out += "\n#SBATCH --partition=ser-par-10g-5"
    out += "\n#SBATCH -N 1"
    out += "\n#SBATCH --workdir=/home/" + personstring + "/wishartbump/workdir"
    out += "\ncd /home/" + personstring + "/wishartbump/"
    out += "\nmpirun -prot -srun -n 1 python train_a3c_gym.py 32 --steps 20000000000 --env cc-"+max_steps+"-v0 --outdir /gss_gpfs_scratch/" + personstring + "/wishart/"+outdir + " --gamma " + str(gam)+ " --eps " + str(eps) + " --nmod " + str(nmod) + " --sigma " + str(sigma) + " --beta " + str(beta)+ " --eval-interval 2000000 --reward-d-pow " + str(power) + " --arch " + arch
            
    f = open("/home/" + personstring + "/wishartbump/scripts/e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+"arch"+arch+".job",'w')
    f.write(out)
    f.close()
    output=subprocess.Popen("sbatch /home/" + personstring +  "/wishartbump/scripts/e"+str(eps)+"k"+str(nmod)+"pow"+str(power)+"beta"+str(beta)+"s"+str(sigma)+"msteps"+max_steps+"arch"+arch+".job",shell=True,stdout=subprocess.PIPE).communicate()[0]
    return output

# June 20 morning runs
 for eps in [1e-50]:
     for sigma in [1e-4]:
         for power in [1]: # used to be 1 3 2
             for nmod in [10,25]:
                 for max_steps in ['10k', '100k']:
                     for beta in [.01,.1,1,10]:
                         for gam in [.9,.95,.99,.999999]:
                             for arch in ["FFSoftmax","LSTMFR"]:
                                 writeScript(eps,nmod,sigma,max_steps,jimstring,beta,power,gam,arch)

# June 4 afternoon runs
# for eps in [1e-50]:
#     for sigma in [1e-4]:
#         for power in [1,3]: # used to be 1 3 2
#             for nmod in [10,25]:
#                 for max_steps in ['10k', '100k']:
#                     for beta in [.01,.1,1,10]:
#                         for gam in [.9,.95,.99,.999999]:
#                             for arch in ["FFSoftmax","LSTMFR"]:
#                                 writeScript(eps,nmod,sigma,max_steps,jimstring,beta,power,gam,arch)

# June 7 afternoon runs
# for eps in [1e-50]:
#     for sigma in [1e-4]:
#         for power in [8]: # used to be 1 3 2
#             for nmod in [10,25]:
#                 for max_steps in ['10k', '100k']:
#                     for beta in [.01,.1,1,10]:
#                         for gam in [.9,.95,.99,.999999]:
#                             for arch in ["FFSoftmax","LSTMFR"]:
#                                 writeScript(eps,nmod,sigma,max_steps,jimstring,beta,power,gam,arch)

# June 7 more afternoon runs
# for eps in [1e-50]:
#     for sigma in [1e-4]:
#         for power in [1,3]: # used to be 1 3 2
#             for nmod in [100]:
#                 for max_steps in ['10k', '100k']:
#                     for beta in [.01,.1,1,10]:
#                         for gam in [.9,.95,.99,.999999]:
#                             for arch in ["FFSoftmax","LSTMFR"]:
#                                 writeScript(eps,nmod,sigma,max_steps,jimstring,beta,power,gam,arch)

# June 8 afternoon runs
#for eps in [1e-50]:
#    for sigma in [1e-4]:
#        for power in [1,3]: # used to be 1 3 2
#            for nmod in [250]:
#                for max_steps in ['10k', '100k']:
#                    for beta in [.01,.1,1,10]:
#                        for gam in [.9,.95,.99,.999999]:
#                            for arch in ["FFSoftmax","LSTMFR"]:
#                                writeScript(eps,nmod,sigma,max_steps,jimstring,beta,power,gam,arch)

# June 11 test precision morning
#for eps in [5e-36]:
#    for sigma in [1e-20]:
#        for power in [1]: # used to be 1 3 2
#            for nmod in [1]:
#                for max_steps in ['10k']:
#                    for beta in [1]:
#                        for gam in [.99]:
#                            for arch in ["FFSoftmax"]:
#                               writeScript(eps,nmod,sigma,max_steps,codystring,beta,power,gam,arch)


#june 17 random walk run
#for eps in [5e-36]:
#    for sigma in [1e-20]:
 #       for power in [0]: # used to be 1 3 2
 #           for nmod in [10]:
 #               for max_steps in ['10k']:
 #                   for beta in [1]:
 #                       for gam in [.99]:
 #                           for arch in ["FFSoftmax"]:
 #                              writeScript(eps,nmod,sigma,max_steps,codystring,beta,power,gam,arch)

# June 8 afternoon runs
#for eps in [1e-50]:
#    for sigma in [1e-4]:
#        for power in [1]: # used to be 1 3 2
#            for nmod in [250]:
#                for max_steps in ['10k', '100k']:
#                    for beta in [100, 1000, 1e4]:
#                        for gam in [.9,.95,.99,.999999]:
#                            for arch in ["FFSoftmax","LSTMFR"]:
#                                writeScript(eps,nmod,sigma,max_steps,jimstring,beta,power,gam,arch)

