'''
#import numpy as np
import random
n = 5
d = 10
z = 10
print(str(n))
print(str(d))
print(str(z))
for i in range(1,n+1):
	num = random.randint(1,5)
	
	for j in range(num):
		word_id = random.randint(1,d)
		freq = random.randint(1, 100)
		print(str(i) + " " + str(word_id) + " " + str(freq))

'''
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize
import pickle

# radius of the circle
circle_r = 10
# center of the circle (x, y)
circle_x = 0
circle_y = 0
data = []

def rand_cluster(n,c,r):
    """returns n random points in disk of radius r centered at c"""
    x,y = c
    points = []
    for i in range(n):
        theta = 2*math.pi*random.random()
        s = r*random.random()
        points.append((x+s*math.cos(theta), y+s*math.sin(theta)))
    return points

def rand_clusters(k,n,r, a,b,c,d):
    """return k clusters of n points each in random disks of radius r
where the centers of the disk are chosen randomly in [a,b]x[c,d]"""
    clusters = []
    for _ in range(k):
        x = a + (b-a)*random.random()
        y = c + (d-c)*random.random()
        clusters.extend(rand_cluster(n,(x,y),r))
    return clusters

data += rand_clusters(1, 700, 0.4, 0,1,0,1)

data += rand_clusters(1, 200, 0.5, 1,2,1,2)

data += rand_clusters(1, 50, 0.9, 0.5, 1.5, 0.5, 1.5)

data = np.array(data)
data_norm = normalize(data, axis=1, norm='l2')

fig = plt.figure()
ax = fig.add_subplot(111)

for i,j in data:
    ax.scatter(i,j,s=50,c='black',marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.show()
plt.show()

with open('xs_norm.pickle', 'wb') as handle:
         pickle.dump(data_norm, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('xs_plot.pickle', 'wb') as handle:
         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


