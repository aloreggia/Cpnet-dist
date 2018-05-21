#from CPNet import CPNet
#a=CPNet()
#a.initFromFile("xml/examples5_alldiff/cpnet_n5c4d2_0014.xml")
#a.getPartialOrder()
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


import numpy as np

file="/home/aloreggi/cpnet/xml/lists/results/examples3/test_NOPRETRAIN_UNBALANCED_DIM_3_21_70_FOLD_0.txt.npy"
print(file)

l=np.load(file)
print(len(l))
print(np.sum(l[:,0]==l[:,1],axis=0))
print(np.sum(abs(l[:,0]-l[:,1]),axis=0))

print("Confusion Matrix:")
print(confusion_matrix(l[:,0], l[:,1]))
print("k-cohen:")
print(cohen_kappa_score(l[:,0], l[:,1]))
print("f1 score:")
print(f1_score(l[:,0], l[:,1],average="micro"))
print("precision score:")
print(precision_score(l[:,0], l[:,1], average="micro"))
print("recall score:")
print(recall_score(l[:,0], l[:,1], average="micro"))

