from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Concatenate, Merge
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, GlobalMaxPooling2D,LSTM
from keras.layers.merge import Dot

from keras.optimizers import RMSprop, Adam


def create_adj_network(input_img, weights=None, train=True):
	'''Base network to be shared (eq. to feature extraction).
	'''
	if weights==None:
		x = Conv2D(8, (3, 3), activation='relu', padding='same', trainable=train)(input_img) #8 3x3
		#x = MaxPooling2D((2, 2), padding='same', trainable=train)(x)
		x = Conv2D(16, (3, 3), activation='relu', padding='same', trainable=train)(x)
		#x = MaxPooling2D((2, 2), padding='same', trainable=train)(x)
		#x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		#encoded = MaxPooling2D((2, 2), padding='same')(x) 
		encoded = Dense(16, activation='relu', trainable=train)(x) #32
	else:
		x = Conv2D(8, (3, 3), activation='relu', padding='same', trainable=train, weights=weights["c1_adj"].get_weights())(input_img) #8 3x3
		#x = MaxPooling2D((2, 2), padding='same', trainable=train, weights=weights["max1_adj"].get_weights())(x)
		x = Conv2D(16, (3, 3), activation='relu', padding='same', trainable=train, weights=weights["c2_adj"].get_weights())(x)
		#x = MaxPooling2D((2, 2), padding='same', trainable=train, weights=weights["max2_adj"].get_weights())(x)
		#x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		#encoded = MaxPooling2D((2, 2), padding='same')(x) 
		encoded = Dense(16, activation='relu', trainable=train, weights=weights["d1_adj"].get_weights())(x) #32
		
	encoded = Flatten()(encoded)
    
	return encoded

def create_cpt_network(input_img, train=True, weights=None ):
	'''Base network to be shared (eq. to feature extraction).
	'''
	if weights==None:
		x = Conv2D(8, (3, 3), activation='relu', padding='same', trainable=train)(input_img) #8 3x3
		#x = MaxPooling2D((2, 2), padding='same', trainable=train)(x)
		x = Conv2D(16, (3, 3), activation='relu', padding='same', trainable=train)(x)
		#x = MaxPooling2D((2, 2), padding='same', trainable=train)(x)
		#x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		#encoded = MaxPooling2D((2, 2), padding='same')(x) 
		encoded = Dense(16, activation='relu', trainable=train)(x) #32
	else:
		x = Conv2D(8, (3, 3), activation='relu', padding='same', trainable=train, weights=weights["c1_cpt"].get_weights())(input_img) #8 3x3
		#x = MaxPooling2D((2, 2), padding='same', trainable=train, weights=weights["max1_cpt"].get_weights())(x)
		x = Conv2D(16, (3, 3), activation='relu', padding='same', trainable=train, weights=weights["c2_cpt"].get_weights())(x)
		#x = MaxPooling2D((2, 2), padding='same', trainable=train, weights=weights["max2_cpt"].get_weights())(x)
		#x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		#encoded = MaxPooling2D((2, 2), padding='same')(x) 
		encoded = Dense(16, activation='relu', trainable=train, weights=weights["d1_cpt"].get_weights())(x) #32
		
	encoded = Flatten()(encoded)
    
	#encoded = LSTM(16)(input_img) #64
	#encoded = Dense(16, activation='relu')(encoded) #32
	
	return encoded
    
	
def create_cptnet_network(input_adjm, input_cpt=None, train=True, weights=None):
	processed_adjm_a = create_adj_network(input_adjm, train=train, weights=weights)
	processed_cpt_a = create_cpt_network(input_cpt, train=train, weights=weights)

	distance = Concatenate()([processed_adjm_a, processed_cpt_a])
	
	return distance
	
	
def buildNN(dim, step, train=True, weights=None):
    #nvidia-smi 
	classes=int(1/step)+1
	classes=1

	# network definition
	input_adjm_a = Input(shape=(dim,dim,1))
	input_cpt_a = Input(shape=(2**dim,dim+1,1))
	
	input_adjm_b = Input(shape=(dim,dim,1))
	input_cpt_b = Input(shape=(2**dim,dim+1,1))

	# because we re-use the same instance `base_network`,
	# the weights of the network
	# will be shared across the two branches
	processed_cpnet1 = create_cptnet_network(input_adjm_a, input_cpt_a, train=train, weights=weights)
	processed_cpnet2 = create_cptnet_network(input_adjm_b, input_cpt_b, train=train, weights=weights)

	distance = Concatenate()([processed_cpnet1, processed_cpnet2])
	
	distance = Dense(1024, activation='relu')(distance)
	distance = Dropout(0.5)(distance)
	distance = Dense(128, activation='relu')(distance)
	distance = Dropout(0.5)(distance)
	distance = Dense(classes, activation='sigmoid')(distance)

	model = Model([input_adjm_a, input_cpt_a, input_adjm_b, input_cpt_b], distance)
	#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	#model.compile(optimizer='Adam', loss='categorical_crossentropy')

	return model
