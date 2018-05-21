import os
import numpy as np
from CPNet import CPNet
from Distances import Distances as D

dir="/home/aloreggi/cpnet/xml/lists/examples6/"
data_path="/home/aloreggi/cpnet/data/examples6/"

start=5
end=10
# File type can be train:
#	Train -- train makes a training set
#	Test -- test makes a test set of files.  -- must run both to create dataset.

file_type="train"

for f in range(start,end):

	print("Processing step: "+str(f))
	file_prefix=dir+"/lists"+file_type+str(f)+".txt"
	order=None

	l=np.loadtxt(file_prefix, dtype="str", delimiter=",")

	list={}
	#list=np.ndarray(list)
	f=open(os.path.join(dir,file_type+str(f)+".txt"),"w")
	print("Number of cpnet in the file:"+str(len(l)))
	for i in range(len(l)-1):
		c=CPNet()
		#print(cpnet)
		c.initFromFile(os.path.join(data_path, l[i]) , oLegalTo=order);
		c.getPartialOrder()
		list[l[i]]=c

	for i in range(len(l)-1):
		for j in range(len(l)-1):
			d = D()
			d.setNets(list[l[i]], list[l[j]], computePO=False)

			kt = d.distKT()
			kt = d.getKTNorm()

			b=l[i]+","+l[j]+","+str(kt)
			f.write(b)
			f.write("\n")
			#list.append(b)

			#print list

	#list = np.asarray(list)
	#np.savetxt(dir+"train0.txt", list, newline="\n")
	f.close()

