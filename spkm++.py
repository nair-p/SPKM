# Spherical K-means plus plus clustering using Lloyd's algorithm in pure Python.
#
# The main program runs the clustering algorithm on a bunch of text documents
#
# To run this code, type the following in the command-line terminal
# python kmeans++.py k test_directory
#
# The program outputs the average cost, the minimum cost, the seeding time and the average running time
# It also outputs the indices of the documents belonging to each of the k clusters

from collections import defaultdict
from math import sqrt
import random
import numpy
import pickle
import sys
import time
import timeit
start_time = time.time()


def dist(x, c):
# This function calculates cosine distance between a given document and a given cluster
    d = 0.
    s = 0
    for i, v in enumerate(x):
        d += (v*c[i])
    return d



def D(x, V):
# This function gives the minimum dissimilarity between a given document and all the cluster centers computed so far
    mx = -100000000000
    for i in V:
       mx = max(mx, dist(x,i))
    return mx


def J(centers, xs):
# This function gives the sum of all the minimum dissimilarties between all documents and cluster centers
    s = 0
    #t = 0

    for i in xs:
        s += D(i, centers)
    return s
   

def probability_dist(xs, V):
# This is the core of the spherical kmeans++ algorithm where new cluster centers are sampled from 
# a probability distibution based on the distance of the points from previously chosen centers
    x_max = []
    j = J(V, xs)
    doc = 1; 
    for x in xs:
        if(j != 0):
           diss = D(x,V)
           x_max.append(diss/j)
           doc += 1
        else:
           print("Check your dataset!")
           return 0
    return x_max


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
    for i in range(l):
        c[i] /= n
    return c

def densify(x, n):
# Convert a sparse vector to a dense one
# A list of the form [(i1,v1), (i2,v2)] becomes [v1,v2]
    d = [0.] * n
    for i, v in enumerate(x):
        d[i] = v
    return d

def cost(xs, centers):
# This function calculates the cost of choosing cluster centers and hence gives cluster quality
    cst = 0
    for x in xs:
        cst += D(x,centers)
    return (len(xs)-cst)


def skmeanspp(k, xs, l, n_iter=100):

    centers = []
    start_time = timeit.default_timer()
    ind = random.sample(range(len(xs)),1)
    centers.append(densify(xs[ind[0]],l))

    for t in range(k-1):
        prob = probability_dist(xs, centers)
        i = numpy.random.choice(len(xs), 1, prob)
        centers.append(densify(xs[i[0]],l))
    
    
    elapsed = timeit.default_timer() - start_time
    cluster = [None] * len(xs)
    t = 0
    start = time.time()
    for i, x in enumerate(xs):
        cluster[i] = max(range(k), key=lambda j: dist(xs[i], centers[j]))
    for j, c in enumerate(centers):
        members = (x for i, x in enumerate(xs) if cluster[i] == j)
        centers[j] = mean(members, l)
    seeding_cost = cost(xs, centers)
    t += (time.time() - start)
    
    print("Seeding cost: %s " % (seeding_cost))
    print("Seeding time: %s seconds " % (elapsed))
    print("Total Running time: %s seconds " % (t))
    
    return cluster


if __name__ == '__main__':

    with open('xs.pickle', 'rb') as handle:
         xs = pickle.load(handle)
    k = [3,5]
    start2 = time.time()
    for i in k:
        print("Number of clusters %d: \n" % i) 
        for p in range(3):
            print("Iteration number %d: \n" % p) 
            cluster_ind = skmeanspp(i, xs, len(xs[0]))
            clusters = [set() for _ in range(i)]
            for q, j in enumerate(cluster_ind):
                clusters[j].add(q)

            for j, c in enumerate(clusters):
                print("cluster %d:" % j)
                #for i in c:
                 #   print("\t%s" % i)
                print(len(c))
                
        print("\n \n") 

    time2 = time.time() - start2
    print("Running time: %s seconds \n" % time2)
