#from CPNet import CPNet
#a=CPNet()
#a.initFromFile("xml/examples5_alldiff/cpnet_n5c4d2_0014.xml")
#a.getPartialOrder()
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


import numpy as np


f1_mean=0.0
k_mean=0.0
mae = 0.0
folds = 10
dim=4

all_results = []
fscores=[]

dir="/home/aloreggi/cpnet/xml/lists/results/examples"+str(dim)+"/"

for i in range(folds):
    file="test_PRETRAIN_NOPOOL_UNBALANCED_DIM_"+str(dim)+"_11_70_FOLD_"+str(i)+".txt.npy"
    #print(file)

    l=np.load(dir+file)
    #print(np.shape(l))
    if all_results == []:
        all_results = l
    else:
        all_results = np.concatenate((all_results,l),axis=0)
    '''
    print(np.sum(l[:,0]==l[:,1],axis=0))
    print(len(l))
    mae += (float(np.sum(abs(l[:,0]-l[:,1]),axis=0)) / len(l))
	'''
    
    print("F1-score: \t" + str(f1_score(l[:,0], l[:,1],average="micro")))
	fscores.append(f1_score(l[:,0], l[:,1],average="micro"))
    print("Cohen k: \t" + str(cohen_kappa_score(l[:,0], l[:,1])))
    
    print("")
    mae += (float(np.sum(abs(l[:,0]-l[:,1]),axis=0)) / len(l))

print("F1-score: \t" + str(f1_score(all_results[:,0], all_results[:,1],average="micro")))
print("Cohen k: \t" + str(cohen_kappa_score(all_results[:,0], all_results[:,1])))
print("MAE: \t\t"+ str(mae / float(folds)))
print("STD: \t\t"+str(np.std(fscores)))
