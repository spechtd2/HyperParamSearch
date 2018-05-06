'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 512
num_classes = 10
epochs = 60

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


Hlayers = [1, 4, 7, 10]
nodes = [10,40,160,640]
#activations = ['elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear' ]
activations = ['elu','softplus','relu','tanh','sigmoid','linear' ]
dropout = [0.1,0.2,0.3,0.4,0.5]
#optimizers = ['SGD', 'RMSprop', 'Adagrad','Adadelta','Adam','Adamax','Nadam']
optimizers = ['SGD', 'RMSprop', 'Adagrad','Adadelta','Adam']
learning = [0.1,0.01,0.001,0.0001]

text_file = open("Output.txt", "w")
text_file.write('File start\n')	
text_file.close() 
						
for k in range(0, len(Hlayers)):
	for l in range(0, len(nodes)):
		for m in range(0, len(activations)):
			for n in range(0, len(dropout)):
				for o in range(0, len(optimizers)):
					for p in range(0, len(learning)):
						#initialize first model and input shape
						model = Sequential()
						model.add(Dense(nodes[l], activation=activations[m], input_shape=(784,)))
						model.add(Dropout(dropout[n]))
						#create a model of appropiate length
						for q in range(0, Hlayers[k]-1):
							model.add(Dense(nodes[l], activation=activations[m]))
							model.add(Dropout(dropout[n]))
		
						# set the output node
						model.add(Dense(10, activation='softmax'))	
						
						if(optimizers[o] == 'SGD'):
							model.compile(loss='categorical_crossentropy',
										optimizer=keras.optimizers.SGD(lr = learning[p]),
										metrics=['accuracy'])
						elif(optimizers[o] == 'RMSprop'):
							model.compile(loss='categorical_crossentropy',
										optimizer=keras.optimizers.RMSprop(lr = learning[p]),
										metrics=['accuracy'])
						elif(optimizers[o] == 'Adagrad'):
							model.compile(loss='categorical_crossentropy',
										optimizer=keras.optimizers.Adagrad(lr = learning[p]),
										metrics=['accuracy'])
						elif(optimizers[o] == 'Adadelta'):
							model.compile(loss='categorical_crossentropy',
										optimizer=keras.optimizers.Adadelta(lr = learning[p]),
										metrics=['accuracy'])
						elif(optimizers[o] == 'Adam'):
							model.compile(loss='categorical_crossentropy',
										optimizer=keras.optimizers.Adam(lr = learning[p]),
										metrics=['accuracy'])
						elif(optimizers[o] == 'Adamax'):
							model.compile(loss='categorical_crossentropy',
										optimizer=keras.optimizers.Adamax(lr = learning[p]),
										metrics=['accuracy'])
						elif(optimizers[o] == 'Nadam'):
							model.compile(loss='categorical_crossentropy',
										optimizer=keras.optimizers.Nadam(lr = learning[p]),
										metrics=['accuracy'])
						else:
							print('That is actually an incorrect answer, how did you do that?')							

						history = model.fit(x_train, y_train,
									batch_size=batch_size,
									epochs=epochs,
									verbose=0,
									validation_data=(x_test, y_test))
						score = model.evaluate(x_test, y_test, verbose=0)
						print(k,l,m,n,o,p)
						text_file = open("Output.txt", "a")
						text_file.write(str(Hlayers[k]) + ' ' + str(nodes[l]) + ' ' + activations[m] + ' ' + str(dropout[n]) + ' ' + optimizers[o] + ' ' + str(learning[p]) + ' ' + str(score[0]) + ' ' + str(score[1])+ '\n')	
						text_file.close() 
						print('Test loss:', score[0])
						print('Test accuracy:', score[1])
