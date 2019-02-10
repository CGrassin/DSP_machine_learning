from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
import matplotlib.pyplot as plt
import random
import math

def randomPeriodicFunction(nbSamples,harmonics,frequency,amplitude=1,noiseAmplitude=0):
	signal = []
	fourierCoef = []

	for n in range(harmonics*2) :
		fourierCoef.append(amplitude*(random.random()-0.5)*2)

	omega = math.pi * 2 * frequency
	for t in range(nbSamples):
		signal.append(noiseAmplitude*random.random())
		for n in range(harmonics):
			signal[-1] += fourierCoef[n]*math.cos(n*t*omega) + fourierCoef[2*n+1]*math.sin(n*t*omega)

	return signal

def corrupt(array,drops):
	for i in range(drops):
		array[random.randint(0,len(array)-1)] = 0
	return array


# ---------TRAINING----------

# Variables
dataPoints = 128

# Generate input/output data
inputs = []
outputs = []
for i in range(10000) :
	inputs.append(randomPeriodicFunction(dataPoints,random.randint(2,5),0.025))
	outputs.append(corrupt(inputs[-1].copy(),50))
inputs = numpy.array(inputs);
outputs = numpy.array(outputs);

# Machine learning
model = Sequential()
model.add(Dense(dataPoints, activation='linear'))
model.add(Dense(dataPoints, activation='linear'))
model.add(Dense(dataPoints, activation='sigmoid'))
model.add(Dense(dataPoints, activation='linear'))
model.add(Dense(dataPoints, activation='linear'))
model.compile(optimizer='rmsprop',loss='mse')
model.fit(inputs, outputs, epochs=30, batch_size=32)
# Export model to PNG
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# ---------PREDICTION---------

# Predict
corruptionRate = 0.5
output = randomPeriodicFunction(dataPoints,4,0.025)
inputK = corrupt(output.copy(),int(corruptionRate*dataPoints))
outputPredict = model.predict(numpy.array([inputK]), batch_size=128)[0]

# Plot
plt.subplot(3, 1, 1)
plt.title("ANN data repair ("+str(dataPoints)+" points, "+str(100*corruptionRate)+"% corruption rate)")
plt.plot(output)
plt.ylabel('Original signal')
plt.subplot(3, 1, 2)
plt.plot(inputK)
plt.ylabel('Input (altered)')
plt.subplot(3, 1, 3)
plt.plot(outputPredict)
plt.ylabel('Output (ANN repair)')
plt.show()