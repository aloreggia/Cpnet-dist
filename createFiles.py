import os
import numpy as np
from CPNet import CPNet
from Distances import Distances as D

dir="/home/aloreggi/cpnet/xml/lists/examples6/"
data_path="/home/aloreggi/cpnet/data/examples6/"

start=0
end=10
# File type can be train:
#	Train -- train makes a training set
#	Test -- test makes a test set of files.  -- must run both to create dataset.

file_type="train"

for j in range(start,end):
	print("Processing step: "+str(j))
	file_prefix="/home/aloreggi/cpnet/xml/lists/examples6/lists"+file_type+str(j)+".txt"
	order=None

	l=np.loadtxt(file_prefix, dtype="str", delimiter=",")

	list=[]
	#list=np.ndarray(list)
	f=open(dir+file_type+str(j)+".txt","w")
	for i in range(len(l)-1):
		for j in range(len(l)-1):
			cpnet1=CPNet()
			cpnet2=CPNet()

			#print l[i]
			#print l[j]

			cpnet1.initFromFile(os.path.join(data_path, l[i]) , oLegalTo=order);
			cpnet2.initFromFile(os.path.join(data_path, l[j]) , oLegalTo=order);
			d = D()
			d.setNets(cpnet1, cpnet2)

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

