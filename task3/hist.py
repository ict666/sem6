import matplotlib.pyplot as plt 
import statistics
import numpy as np
from math import log

file = open("./accuracy/accuracy_eps/accuracy260001.txt", "r")
x = file.read().split("\n")
del x[-1]
file.close()

my_lst = []
for item in x:
    my_lst.append(float(item))
avg = statistics.mean(my_lst)
print(avg)

# An "interface" to matplotlib.axes.Axes.hist() method
plt.style.use('ggplot')
n, bins, patches = plt.hist(x=my_lst, bins=20, color='#2E8B57',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('1 - F (LOSS)')
plt.ylabel('Frequency')
plt.title('n = 26, e = 0.001')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
#plt.savefig('./hists/24.png')
#plt.hist(x, bins = 10)
#plt.show()