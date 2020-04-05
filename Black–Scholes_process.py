#!/usr/bin/env python
# coding: utf-8

# # Plot the Black–Scholes process  
# Refarence1 : https://www.youtube.com/watch?v=mrExmReKrcM&list=PLhDAH9aTfnxIhf-iRKYTVOSXPqDGgfRFP&index=4  
# Refarence2 : https://en.wikipedia.org/wiki/Geometric_Brownian_motion 

import numpy as np
import matplotlib.pyplot as plt

# Define the constant
a = 1
b = 1
loc = 0
scale = 1
delta_t = 0.0001
sqrt_delta_t = np.sqrt(delta_t)
loop_length = int(1/delta_t)

# Create time step array
time = np.arange(loop_length)

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# Black–Scholes process
for i in range(20):
    X = []
    X.append(1)
    w = []
    w.append(0)
    for t in range(1, loop_length):
        delta_w = np.random.choice([-1,1], 1) * scale * sqrt_delta_t
        w.append(w[t-1] + delta_w)
        X.append (X[0] * np.exp((a - (1/2) * (b ** 2) * (scale ** 2)) * (t * delta_t) + b * w[t]))
        
    ax.plot(time * delta_t, X, lw = 0.5)
    
# Add non-noise process 
b_base = 0
X = []
X.append(1)
w = []
w.append(0)
for t in range(1, loop_length):
    delta_w = np.random.choice([-1,1], 1) * scale * sqrt_delta_t
    w.append(w[t-1] + delta_w)
    X.append (X[0] * np.exp((a - (1/2) * (b_base ** 2) * (scale ** 2)) * (t * delta_t) + b_base * w[t]))

ax.plot(time* delta_t, X)

# plot
plt.title('Black–Scholes process: b = 1')
plt.xlabel('Time')
plt.ylabel('x')
plt.hlines(y=1, xmin=0, xmax=1, colors='black', lw = 2)
plt.ylim(-0.05, 10)
plt.xlim(0, 1)
plt.grid(True)

# plt.show()
plt.savefig('Black–Scholes_process.png')

# Brownian motion (Random walk)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

for i in range(20):
    w = []
    w.append(0)
    for t in range(1, loop_length):
        delta_w = np.random.choice([-1,1], 1) * scale * sqrt_delta_t
        w.append(w[t-1] + delta_w)

    ax.plot(time* delta_t, w , lw = 0.5)

# plot
plt.title('Brownian motion (Random walk)')
plt.xlabel('Time')
plt.ylabel('x')
plt.hlines(y=0, xmin=0, xmax=1, colors='black', lw = 2)
plt.ylim(-5, 5)
plt.xlim(0, 1)
plt.grid(True)

# plt.show()
plt.savefig('Brownian_motion.png')




