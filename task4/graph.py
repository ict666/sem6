import matplotlib.pyplot as plt
  
# x axis values
y = [1, 1.833512, 3.131565, 4.373336]
# corresponding y axis values
x = [32, 64, 128, 256]
  
# plotting the points 
plt.plot(x, y)
  
# naming the x axis
plt.xlabel('Number of procs')
# naming the y axis
plt.ylabel('Speedup')
  
# giving a title to my graph
plt.title('Speedup for CNOT')
plt.savefig("cnot.png")
  
# function to show the plot
plt.show()