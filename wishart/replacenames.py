import os
import sys
import sys

workdir = sys.argv[1]
ii = 0
for dir in os.listdir(workdir):
	#print workdir  + dir + '/' + "command.txt"
	if os.path.isfile(workdir  + dir):
		f = open(workdir + dir,'r')
		dats = f.read()
		f.close
		#print dats
		#dats = dats.replace('jhhalverson','codylong')
		newdats = dats.replace('/home/codylong/wishart/inout/','/gss_gpfs_scratch/codylong/wishart/inout/')
		#newdats = newdats.replace('output_cc_prec','new_origin_new_reward')
		#newdats = newdats + '\''
		#newdats = newdats.replace('\n\'','\'').replace(' --arch', '\' --arch').replace('FR\'','FR')
		#newdats = newdats.replace('origin ','origin \'')
		#newdats = newdats.replace('\'-','\' -')
		#newdats = newdats.replace('new','old').replace('.job','2.job').replace('.err','2.err').replace('.out','2.out')
		f2  = open(workdir + dir,'w')
		f2.write(newdats)
		f2.close
print ii
