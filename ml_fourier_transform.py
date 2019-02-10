from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
import matplotlib.pyplot as plt
import random
import math
import time
import statistics

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

# -------------------

# Variables
dataPoints = 256

# Array generation
functions = []
ffts = []
for i in range(5000) :
	functions.append(randomPeriodicFunction(dataPoints,20,0.1))
	ffts.append(numpy.fft.rfft(functions[-1]))
functions = numpy.array(functions);
ffts = numpy.array(ffts);

# Machine learning
model = Sequential()
model.add(Dense(dataPoints, activation='linear'))
model.add(Dense(int(dataPoints/2)+1, activation='linear'))
model.compile(optimizer='rmsprop',loss='mse')
model.fit(functions, ffts, epochs=10, batch_size=32)

# -------------------

# Predict
inputK = randomPeriodicFunction(dataPoints,5,0.1)
outputCalc = numpy.fft.rfft(inputK)
outputPredict = model.predict(numpy.array([inputK]), batch_size=128)[0]

# Plot
plt.subplot(3, 1, 1)
plt.plot(inputK)
plt.ylabel('Input')
plt.title('Predicted vs. FFT computation')
plt.subplot(3, 1, 2)
plt.plot(outputCalc)
plt.ylabel('FFT')
plt.subplot(3, 1, 3)
plt.plot(outputPredict)
plt.ylabel('ML prediction')
plt.show()

# -------------------------

# Stats

# times=[]
# for i in range(1000):  
#     t0 = time.time()
#     numpy.fft.rfft(inputK[-1])
#     t1 = time.time()
#     times.append((t1 - t0) * 1000)

# print('FFT ', len(times), 'times')
# print('\tMEDIAN', statistics.median(times))
# print('\tMEAN  ', statistics.mean(times))
# print('\tSTDEV ', statistics.stdev(times))
