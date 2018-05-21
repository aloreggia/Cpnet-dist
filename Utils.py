import os
from CPNet import CPNet
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def createRandomTriples(file, nTriples=1000):
	listFiles=np.genfromtxt(file,delimiter=",",dtype="unicode")
	listFiles=np.delete(listFiles, len(listFiles)-1)
	
	randIndex=np.random.randint(0,high=len(listFiles)-1, size=(nTriples,3))
	
	listTriples=[]
	for i in randIndex:
		listTriples.append((listFiles[i[0]],listFiles[i[1]],listFiles[i[2]]))
		
	return np.array(listTriples)
		

#from Distances import Distances as D
def get_one_hot(targets, nb_classes):
	return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def getOLegal(dir, order, verbose=False):
	
	listOLegal=[]
	
	for file in os.listdir(dir):
		if file.endswith(".xml"):
			
			temp=CPNet();
			temp.initFromFile(os.path.join(dir, file));
			if temp.isOlegal(order): 
				if verbose: print(os.path.join(dir, file))
				listOLegal.append(temp)
				
	return listOLegal;
	
def getList(dir, order, verbose=False):
	
	listOLegal=[]
	
	for file in os.listdir(dir):
		if file.endswith(".xml"):
			
			temp=CPNet();
			temp.initFromFile(os.path.join(dir, file), oLegalTo=order);
			listOLegal.append(temp)
				
	return listOLegal;
	
def getEqual(dir, order, verbose=False):
	listEqual=[]
	
	listFiles=os.listdir(dir)
	i=0
	n=len(listFiles)
	
	for i in range(n):
		temp1=CPNet();
		temp1.initFromFile(os.path.join(dir, listFiles[i]), oLegalTo=order);
		for j in range(i+1,n):
			
			temp2=CPNet();
			temp2.initFromFile(os.path.join(dir, listFiles[j]), oLegalTo=order);
			
	
def make_dataset(file, take_per_class=1000, classes=10, prefix="train"):

	"""
	Build a balanced dataset. For each class take a number of samples specifiec from take_per_class
	"""
	dataset = np.genfromtxt(file,delimiter=",",dtype="S50")
	dataset = shuffle(dataset, random_state=42)
	step= 1. / classes
	
	#Build a dictionary of classes to which there still samples to add
	to_add = {i:0 for i in range(0,classes)}
	to_write=[]
	
	for p in dataset:
		key = float(p[2])
		key = int(key/step)
		
		if key in to_add.keys(): 
			to_add[key] += 1
			to_write.append(p)
			#When the class has the specified number of samples, remove it from the dict, so no more samples will be added
			if to_add[key] == take_per_class: del to_add[key]
			if len(to_add) == 0: 
				np.savetxt(prefix+str(take_per_class)+"_"+str(classes)+".txt", to_write, newline="\n", fmt="%s")
				return
			
	np.savetxt(prefix+str(take_per_class)+"_"+str(classes)+".txt", to_write, newline="\n", delimiter=",", fmt="%s")
	
def cpnet_distribution(file, step=0.1, take=None):
	classes=int(1/step)+1
	take=None
	file="xml/lists/examples3/train0.txt"

	dataset = np.genfromtxt(file,delimiter=",",dtype="S50", max_rows=take)
	dataset = shuffle(dataset, random_state=42)

	print(len(dataset))
	dataset_label = np.array(dataset[:,2].astype("f4") /step, dtype="i2")

	print(len(dataset))

	y = np.zeros(classes)
	for i in dataset_label:
		#print i
		y[i] = y[i] + 1

	return y,dataset

	

def getDir(r1,o1,o2):
	s=""
	if r1==-1:
		s= o1 + " -- " + o2
	if r1==0:
		s= o1 + " -> " + o2
	if r1==1:
		s= o1 + " <- " + o2
	
	return s;

def load_file(file, take=None, single=False, startIndex=0, test_size=0.2):
	'''
	Load data from file and return splitted dataset
	'''
	dataset = np.genfromtxt(file,delimiter=",",dtype="unicode", max_rows=take)
	dataset = shuffle(dataset, random_state=42)
	#print dataset
	#If spedcified take the first n record
	#if take!=None:
	#	dataset=np.take(dataset,[i for i in range(startIndex,startIndex+take)], axis=0)
	
	if test_size==None: return dataset[:,0], dataset[:,1], dataset[:,2]
	
	train,test = train_test_split(dataset, test_size=test_size, random_state=42)
	#print train
	
	return (train[:,0], train[:,1], train[:,2]), (test[:,0], test[:,1], test[:,2])

def getData(list_samples, prefix="", dim=5):
	
	#trainx1=np.ndarray(shape=(len(list_samples),dim,dim))
	list={}
	for cpnet in list_samples:
		list[cpnet]=None

	for key in list:
		temp=CPNet()
		#print(key)
		temp.initFromFile(prefix+str(key))
		list[key]=temp
	
	trainx1=[]
	trainx2=[]
	
	for i in range(len(list_samples)):
		#temp = CPNet()
		#print prefix+list_samples[i]
		#temp.initFromFile(prefix+list_samples[i])
		trainx1.append(list[list_samples[i]].getAdjMatrix())
		trainx2.append(list[list_samples[i]].getCPTList())
        #print trainx1[i]
	
	
	#print "Shape trainx2 pre: "
	#print trainx2.shape
	
	trainx1=np.asarray(trainx1)
	trainx2=np.asarray(trainx2)
	#print "Shape trainx2: "
	#print trainx2.shape
	return trainx1,trainx2
	
def load_data(file, take=50000, dim=5, prefix="", startIndex=0, batch_size=128, step=0.05, single=False):
	print( file)
	print( take)
	
	print( "Starting loading data ")

	(temp_trainx1, temp_trainx2, temp_trainlab) = load_file(file, take=batch_size, startIndex=0, test_size=None)

	temp_trainlab = temp_trainlab.astype("f4")
	##temp_trainlab = np.array(temp_trainlab / step, dtype="i2")
	trainlab = np.reshape(temp_trainlab,(-1,1))
	#print( temp_trainx2)
	##trainlab = get_one_hot(temp_trainlab, int(1/step)+1)
	#print temp_trainx2
	#testlab=temp_testlab.astype("f4")
	
	#testxa,testxb=getData(temp_testx1, prefix, dim)

	#print "getData 2: "
	#print temp_trainx2
	trainxc,trainxd=getData(temp_trainx2, prefix, dim)
	#print trainxd.shape

	if single: 
			return (trainxc,trainxd)	
	
	#print "getData 1: "	
	#print temp_trainx1
	trainxa,trainxb=getData(temp_trainx1, prefix, dim)
	#print trainxb.shape
	
	#trainxa,_,_ = reshape_data(trainxa, dimx=dim, dimy=dim)
	#trainxb,_,_ = reshape_data(trainxb, dimx=2**dim, dimy=dim+1)
	#trainxc,_,_ = reshape_data(trainxc, dimx=dim, dimy=dim)
	#trainxd,_,_ = reshape_data(trainxd, dimx=2**dim, dimy=dim+1)
	
	#return (trainx1[:,0],trainx1[:,1],trainx2[:,0],trainx2[:,1],trainlab),(testx1[:,0],testx1[:,1],testx2[:,0],testx2[:,1],testlab)
	#print(trainlab)
	return (trainxa,trainxb,trainxc,trainxd,trainlab)

def load_large_train(file, take=50000, dim=5, prefix="", startIndex=0, batch_size=128):
	print( file)
	print( take)
	
	n=take/batch_size
	
	while True:
		for j in range(n):
			print( "Starting train index: " + str(j))
			
			(temp_trainx1, temp_trainx2, temp_trainlab),(_,_,_) = load_file(file, take=batch_size, startIndex=j*batch_size, test_size=0.2)
		
		
			trainx1=getData(temp_trainx1, prefix, dim)
			trainx2=getData(temp_trainx2, prefix, dim)
			#trainlab=np.ndarray(temp_trainlab)
			trainlab=temp_trainlab.astype("f4")
			
			#trainx1,_,_ = reshape_data(trainx1, meanCanc=False)
			#trainx2,_,_ = reshape_data(trainx2, meanCanc=False)
			
			#trainx = np.concatenate(trainx1,trainx2)

		#print "Input shape: "
		#trainx = [trainx1[:,0],trainx1[:,1],trainx2[:,0],trainx2[:,1]]
		#trainx = np.asarray(trainx)
		#print trainx.shape
		yield trainx1[:,0],trainx1[:,1],trainx2[:,0],trainx2[:,1],trainlab

def load_large_test(file, take=50000, dim=5, prefix="", startIndex=0, batch_size=128):
	print( file)
	print( take)
	
	n=take/batch_size
	
	while True:
		for j in range(n):
			print( "Starting test index: " + str(j))
			
			(_,_,_),(temp_testx1, temp_testx2, temp_testlab) = load_file(file, take=batch_size, startIndex=j*batch_size, test_size=0.2)
		
		
			testx1=getData(temp_testx1, prefix, dim)
			testx2=getData(temp_testx2, prefix, dim)
			#testlab=np.ndarray(temp_testlab)
			testlab=temp_testlab.astype("f4")
			
			#testx1,_,_ = reshape_data(testx1, meanCanc=False)
			#testx2,_,_ = reshape_data(testx2, meanCanc=False)
			

			yield testx1[:,0],testx1[:,1],textx2[:,0],textx2[:,1],testlab

			
			
def reshape_data(arr, dimx=5, dimy=5, reshape=True, norm=False, maxValue=255.0, tempMean=[], tempSTD=None, meanCanc=True):
	if norm==True: arr = arr.astype('float32') / maxValue
	else: arr = arr.astype('float32')
	
	if reshape: arr = np.reshape(arr, (len(arr), dimx, dimy, 1))  # adapt this if using `channels_first` image data format
	#print arr
	
	#print tempMean
	if meanCanc==True:
		if tempMean==[]:
			tempMean = np.mean(arr, axis=0)
			tempSTD = np.std(arr, axis = 0)
			tempSTD = tempSTD + 1e-8
			
		arr = (arr - tempMean)/tempSTD
		
		return arr, tempMean, tempSTD
	
	return arr,None,None
