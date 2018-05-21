import numpy as np
import os

np.random.seed(1234)

import random


from keras import optimizers
from keras import backend as K
from sklearn import cross_validation
import Utils as u

import sys, getopt

from keras.models import model_from_json
from buildNN import buildNN

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from sklearn.metrics import r2_score

import time
	
def main(argv):
    #nvidia-smi 
	dim = 3
	step=1
	classes=int(1/step)+1
	epochs = 30
	nfolds=10
	#if take=None then take all the examples
	
	
	only_test=True
	
	data_path="/Users/aloreggia/Dropbox/universita/grant/ethics/distanza/codice/xml/examples"+str(dim)+"_alldiff/"
	prefix_results= "/Users/aloreggia/Dropbox/universita/grant/ethics/distanza_old_codice/results/examples"+str(dim)+"/"
	list_path="/Users/aloreggia/Dropbox/universita/grant/ethics/distanza/codice/xml/lists/examples"+str(dim)+"/"
	batch_size = 128
	
	
	trainable=True
	f=open(os.path.join(prefix_results,"regression_PRETRAIN_"+str(dim)+". txt"),"w")
	for j in range(0,nfolds):
		
		take = None
		samples_per_epoch = take
		
		
		json_file_adj= prefix_results+"autoencoder_NOPOOL_adj"+str(j)+".json"
		weights_file_adj= prefix_results + "autoencoder_NOPOOL_adj"+str(j)+".h5"
		json_file_cpt= prefix_results+"autoencoder_NOPOOL_cpt"+str(j)+".json"
		weights_file_cpt= prefix_results + "autoencoder_NOPOOL_cpt"+str(j)+".h5"
		
		file_training=list_path+"train"+str(j)+".txt"
		file_test=list_path+"test"+str(j)+".txt"
		#outputfile="_PRETRAIN_NOTRAIN-CNN_BALANCED_DIM_"+str(dim)+"_"+str(classes)+"_"+str(take)+"_"+str(epochs)+"_"
		outputfile="_REGRESSION_PRETRAIN_DIM_"+str(dim)+"_"+str(classes)+"_"+str(epochs)+"_FOLD_"+str(j)
		file_weight= prefix_results + "weights."+outputfile+".best.hdf5"

		#millis = int(round(time.time() * 1000))
		
		json_file_adj = open(json_file_adj, 'r')
		loaded_model_json_adj = json_file_adj.read()
		json_file_adj.close()
		loaded_model_adj = model_from_json(loaded_model_json_adj)
		loaded_model_adj.load_weights(weights_file_adj)
		layer_dict={}
		for layer in loaded_model_adj.layers:
			layer_dict[layer.name]=layer

		json_file_cpt = open(json_file_cpt, 'r')
		loaded_model_json_cpt = json_file_cpt.read()
		json_file_cpt.close()
		loaded_model_cpt = model_from_json(loaded_model_json_cpt)
		loaded_model_cpt.load_weights(weights_file_cpt)

		for layer in loaded_model_cpt.layers:
			layer_dict[layer.name]=layer
		
		#layer_dict = dict([(layer.name, layer) for layer in loaded_model_adj.layers])
		
		#print(file_weight)
		#layer_dict=None
		model = buildNN(dim, step, train=trainable, weights=layer_dict)
		model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
		print( model.summary())
		
		millis = int(round(time.time() * 1000))
		# the data, shuffled and split between train and test sets
		(trainx1,trainx2,trainx3,trainx4,trainlab) = u.load_data(file_training,take=take, dim=dim, step=step, prefix=data_path,batch_size=samples_per_epoch)

		take=len(trainx1)	

		print("Shape: ")
		print(trainx2.shape)
		trainx1=np.reshape(trainx1,(take, dim*dim))
		trainx2=np.reshape(trainx2,(take, (2**dim)*(dim+1)))
		trainx3=np.reshape(trainx3,(take, dim*dim))
		trainx4=np.reshape(trainx4,(take, (2**dim)*(dim+1)))
		print(trainx2.shape)

		scaler1 = StandardScaler().fit(trainx1)
		scaler2 = StandardScaler().fit(trainx2)
		scaler3 = StandardScaler().fit(trainx3)
		scaler4 = StandardScaler().fit(trainx4)


		trainx1 = scaler1.transform(trainx1)
		trainx2 = scaler2.transform(trainx2)
		trainx3 = scaler3.transform(trainx3)
		trainx4 = scaler4.transform(trainx4)


		trainx1=np.reshape(trainx1,(take, dim,dim,1))
		trainx2=np.reshape(trainx2,(take, (2**dim),(dim+1),1))
		trainx3=np.reshape(trainx3,(take, dim,dim,1))
		trainx4=np.reshape(trainx4,(take, (2**dim),(dim+1),1))

		checkpoint = ModelCheckpoint(file_weight, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]

		model.fit([trainx1,trainx2,trainx3,trainx4],y=trainlab,
					epochs=epochs,
					batch_size=batch_size,
					shuffle=True,
					validation_split=0.2, 
					callbacks=callbacks_list)
					#validation_data=((testx1,testx2,testx3,testx4),testlab), callbacks=callbacks_list)


		millis = int(round(time.time() * 1000)) - millis
		'''
		model.load_weights(file_weight)
		tr_acc = model.predict([trainx1,trainx2,trainx3,trainx4])
		#np.save(prefix_results + "train_output_last_layer"+outputfile+".txt",tr_acc)
		#print tr_acc.shape
		#print tr_acc
		#print trainlab.shape
		#print trainlab
		tr_acc = np.argmax(tr_acc, axis=1)
		tr_acc = np.reshape(tr_acc,(len(tr_acc),1))

		#trainlab = np.reshape(trainlab,(len(trainlab),1))
		trainlab = np.argmax(trainlab, axis=1)
		trainlab = np.reshape(trainlab,(len(trainlab),1))
		#te_acc = model.predict([x1_test, x2_test])

		#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc[1]))
		#np.save(prefix_results + "train"+outputfile+".txt",np.concatenate((trainlab,tr_acc),axis=1))
		#print('* Accuracy on test set: %0.2f%%' % (100 * te_acc[1]))
		#np.save("test"+outputfile+".txt",np.concatenate((y_test,te_acc),axis=1))
		'''
		
		model.load_weights(file_weight)
			
		(trainx1,trainx2,trainx3,trainx4,trainlab) = u.load_data(file_test,take=take, dim=dim, step=step, prefix=data_path,batch_size=samples_per_epoch)
		#print("TEST DATASET LABELS")
		#print(trainlab)
		take=len(trainx1)
		
		trainx1=np.reshape(trainx1,(take, dim*dim))
		trainx2=np.reshape(trainx2,(take, (2**dim)*(dim+1)))
		trainx3=np.reshape(trainx3,(take, dim*dim))
		trainx4=np.reshape(trainx4,(take, (2**dim)*(dim+1)))
		
		trainx1 = scaler1.transform(trainx1)
		trainx2 = scaler2.transform(trainx2)
		trainx3 = scaler3.transform(trainx3)
		trainx4 = scaler4.transform(trainx4)
		
		trainx1=np.reshape(trainx1,(take, dim,dim,1))
		trainx2=np.reshape(trainx2,(take, (2**dim),(dim+1),1))
		trainx3=np.reshape(trainx3,(take, dim,dim,1))
		trainx4=np.reshape(trainx4,(take, (2**dim),(dim+1),1))
		
		
		tr_acc = model.predict([trainx1,trainx2,trainx3,trainx4])
		#np.save(prefix_results + "test_output_last_layer"+outputfile+".txt",tr_acc)
		#print tr_acc.shape
		#print tr_acc
		#print trainlab.shape
		#print trainlab
		#tr_acc = np.argmax(tr_acc, axis=1)
		tr_acc = np.reshape(tr_acc,(len(tr_acc),1))
		
		mse_value, mae_value = model.evaluate([trainx1,trainx2,trainx3,trainx4], trainlab, verbose=0)
		r2=r2_score(trainlab, tr_acc)
		
		f.write("MAE: " + str(mae_value)+" R2: "+str(r2)+"\n")
		
		
		#trainlab = np.reshape(trainlab,(len(trainlab),1))
		#trainlab = np.argmax(trainlab, axis=1)
		trainlab = np.reshape(trainlab,(len(trainlab),1))
		#te_acc = model.predict([x1_test, x2_test])
		#print(np.concatenate((trainlab,tr_acc),axis=1))
		#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc[1]))
		np.save(prefix_results + "test"+outputfile+".txt",np.concatenate((trainlab,tr_acc),axis=1))

	
if __name__ == "__main__":
	main(sys.argv[1:])

