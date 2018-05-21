import os
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import numpy as np

dir="/home/aloreggi/cpnet/data/examples6"
file_prefix="/home/aloreggi/cpnet/xml/lists/examples6/"
nFolds=10


listFiles=os.listdir(dir)
listFiles=np.array(listFiles)

kf = KFold(len(listFiles), nFolds, shuffle = True)
count = 0 
for train, test in kf:
	print("Fold " + str(count))
	tempTrain = listFiles[train] 
	tempTest = listFiles[test] 
	np.savetxt(file_prefix+"liststrain"+str(count)+".txt", tempTrain, newline=",", fmt="%s")
	np.savetxt(file_prefix+"liststest"+str(count)+".txt", tempTest, newline=",", fmt="%s")
	count += 1
