import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Dense
import numpy

#fix random seed for reproducibility
numpy.random.seed(7)

#load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
#split into input(X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8] #the dataset has 9 columns

#create the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
model.fit(X, Y, epochs=150, batch_size=10)

#evaluate the mode
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#make preductions!!!
predictions = model.predict(X)
#round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
