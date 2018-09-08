# Spherical K-means clustering using Lloyd's algorithm in pure Python.
#
# To run this code, type the following in the command-line terminal
# python spherical_KMeans.py
#
# The program outputs the average cost, the minimum cost, the seeding time and the average running time
# It also produces the indices of the documents belonging to each of the k clusters

from collections import defaultdict
from math import sqrt
import random
import math
import pickle
import time
import timeit
start_time = time.time()


def densify(x, n):
# Convert a sparse vector to a dense one
    d = [0.] * n
    for i, v in enumerate(x):
        d[i] = v
    return d


def dist(x, c):
# This function calculates cosine distance between a given document and a given cluster
    d = 0.
    for i, v in enumerate(x):
        d += (v*c[i])
    return 1.5 - d

def D(x, V):
# This function gives the minimum dissimilarity between a given document and all the cluster centers computed so far
    mn = 100000000
    for i in V:
       mn = min(mn, dist(x,i))
    return mn

def mean(xs, l):
# Mean (as a dense vector) of a set of sparse vectors of length l
    c = [0.] * l
    t = [0.] * l
    n = 0
    for x in xs:
        for i, v in enumerate(x):
            c[i] += v
        n += 1
    if( n == 0):
        return t

    den = 0.
    for i in range(l):
        c[i] /= n
        den += (c[i]*c[i])
    for i in range(l):
        c[i] /= den
    return c

def cost(xs, centers):
# This function calculates the cost of choosing cluster centers and hence gives cluster quality
    cst = 0
    for x in xs:
        cst += D(x,centers)
    return cst

def kmeans(k, xs, l, n_iter=3):
    # Initialize from random points.
    start_time = timeit.default_timer()
    centers = [densify(xs[i], l) for i in random.sample(list(range(len(xs))), k)]
    elapsed = timeit.default_timer() - start_time
    cluster = [None] * len(xs)
    t = 0
    count = 0
    current = 2
    prev = 1
    maxi = 10000000
    avg = 0
    while( count < n_iter ):
         start = time.time()
         count += 1
         for i, x in enumerate(xs):
             cluster[i] = min(range(k), key=lambda j: dist(xs[i], centers[j]))
         for j, c in enumerate(centers):
             members = (x for i, x in enumerate(xs) if cluster[i] == j)
             centers[j] = mean(members, l)
        

         maxi = min(maxi,cost(xs, centers))
         t += (time.time() - start)
    
  
    print("Seeding cost: %s " % (maxi))
    print("Seeding time: %s seconds " % (elapsed))
    print("Total Running time: %s seconds " % (t))
    
    
    return cluster


if __name__ == '__main__':

    with open('xs.pickle', 'rb') as handle:
         xs = pickle.load(handle)
    k = [3, 5]
    start = time.time()
    for i in k:
        print("Number of clusters %d: \n" % i) 
        for p in range(3):
            print("Iteration number %d: \n" % p) 
            cluster_ind = kmeans(i, xs, len(xs[0]))
            clusters = [set() for _ in range(i)]
            for q, j in enumerate(cluster_ind):
                 clusters[j].add(q)

        for j, c in enumerate(clusters):
            print("cluster %d:" % j)
	    print(len(c))
	print("\n \n")

    time2 = time.time() - start
    print("Running time: %s seconds \n" % time2)
