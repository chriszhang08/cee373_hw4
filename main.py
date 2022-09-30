import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# %%
# open river.csv and use that data to create a histogram
open('river.csv', 'r')
data = np.genfromtxt('river.csv', delimiter=',')

# %%
num_bins = 500

# construct a freqeuncy histogram of the data
# 500 equal-width bins
n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)
# add axes and title
plt.xlabel('Size of river flow in cfs')
plt.ylabel('Frequency')
plt.title('River flows since 2010')
plt.show()

# %%
# calculate mean and sample standard deviation
mean = np.mean(data)
std = np.std(data)
print('mean = ', mean)
print('standard deviation = ', std)

#%%
# what is the probability that a river flow is greater than 1000 cfs?
# count how many values are greater than 1000
count = 0
for i in range(len(data)):
    if data[i] > 1000:
        count += 1
# calculate the probability
prob = count / len(data)
print('probability = ', prob)
