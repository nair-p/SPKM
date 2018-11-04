# Spherical K-means MCMC clustering using Lloyd's algorithm in pure Python.
#
# The main program runs the clustering algorithm on a bunch of text documents
# The directory containing these text documents must be given as a command-line argument
# These documents are first converted to sparse vectors, represented as lists of (index, value) pairs.
# The text document must be in the following form :
# No of documents
# No of words in vocabulary
# Total number of words
# Doc_index Word_index frequency
#
# k is the number of clusters 
# test_directory is the name of the directory containing the text corpus
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
import matplotlib.pyplot as plt
start_time = time.time()


def dist(x, c):
# This function calculates cosine distance between a given document and a given cluster
    d = 0.
    for i, v in enumerate(x):
        d += (v*c[i])
    return 1.5 - d


def D(x, V):
# This function gives the minimum dissimilarity between a given document and all the cluster centers computed so far
    mx = 100000000000
    for i in V:
        mx = min(mx, 1-dist(x,i))
    return mx
    '''mx = -1000000000000
    for i in V:
        mx = max(mx, dist(x,i))
    return mx'''
    
def D2(x, V):
    mx = -1000000000000
    for i in V:
        mx = max(mx, dist(x,i))
    return mx

def J(centers, xs):
# This function gives the sum of all the minimum dissimilarties between all documents and cluster centers
    s = 0
    for i in xs:
        s += D(i, centers)
    return s
   

def probability_dist(xs, V):
# This is the core of the spherical kmeans++ algorithm where new cluster centers are sampled from 
# a probability distibution based on the distance of the points from previously chosen centers
    x_max = []
    j = J(V, xs)
    for x in xs:
        if(j != 0):
           x_max.append((D(x,V)/(2 * j)))
        else:
           print("Check your dataset!")
           return 0
    return x_max


def densify(x, n):
# Convert a sparse vector to a dense one
# A list of the form [(i1,v1), (i2,v2)] becomes [v1,v2]
    d = [0.] * n
    for i, v in enumerate(x):
        d[i] = v
    return d


def normalise(xs):
# This function normalises the points in the document corpus so that they can be represented in a unit sphere
    for j in xrange(len(xs)):
        s = 0
        for i, v in xs[j]:
            s += (v*v)
        s = sqrt(s)
        
        for k in xrange(len(xs[j])):
            
            ind = xs[j][k][0]
            freq = xs[j][k]/s
            xs[j][k] = freq

    return xs

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
    for i in xrange(l):
        c[i] /= n
    return c


def cost(xs, centers):
# This function calculates the cost of choosing cluster centers and hence gives cluster quality
    cst = 0
    for x in xs:
        cst += D(x,centers)
    return cst

def plot_graph(cluster_ind, i, iter):

    clusters = [set() for _ in range(i)]
    for q, j in enumerate(cluster_ind):
        clusters[j].add(q)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #scatter = ax.scatter(x,y,c=Cluster,s=50)
    cluster1 = data[list(clusters[0])]
    cluster2 = data[list(clusters[1])]
    for i,j in cluster1:
        ax.scatter(i,j,s=50,c='red',marker='.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #plt.colorbar(scatter)
    for i,j in cluster2:
        ax.scatter(i,j,s=50,c='green',marker='.')
    ax.set_title("no of steps=" + str(iter))
    fig.show()
    plt.show()

def skmeanspp(k, xs, l, n_iter=1000):

    centers = []
    current = 2
    prev = 1
    avg = 0
    minimum = 100000000
    count = 0
    start_time = timeit.default_timer()
  
    #select first centre randomly
    ind = random.sample(xrange(len(xs)),1)
    centers.append(densify(xs[ind[0]],l))
    
    #list of chosen centers
    chosen_centers = []
    chosen_centers.append(ind[0])
    
    #generate probability distribution
    prob_temp = probability_dist(xs, centers)
    
    prob = []
    #add mixing term to probability distribution
    for i in prob_temp:
    	i = i + 1.0/(2.0 * len(xs))
    	prob.append(i)
   
    cum_prob = []
    cum_prob.append(prob[0])
    for i in range(1, len(prob)):
        cum_prob.append(cum_prob[i-1] + prob[i])
    
    for t in range(k-1):
        '''uni_rand = numpy.random.uniform(0,1,1)
        for j,p in enumerate(cum_prob):
            if(uni_rand < p):
                x = j
                break'''
       	x = -1
        while(1):
            cnt = 0
            uni_rand = numpy.random.uniform(0,1,1)
            for j,p in enumerate(cum_prob):
                if(uni_rand < p and j not in chosen_centers):
                    x = j
                    cnt = 1
                    break
                elif(uni_rand < p and j in chosen_centers):
                    cnt = 0
                    break
            if(cnt == 1):
                break
            else:
                continue
	   
    	dx = D(xs[x], centers)
    	
    	for j in range(2,m+1):
            y = -1
            while(1):
            	cnt = 0
            	uni_rand = numpy.random.uniform(0,1,1)
            	for j,p in enumerate(cum_prob):
            	    if(uni_rand < p and j not in chosen_centers):
                        y = j
                        cnt = 1
                    	break
                    elif(uni_rand < p and j in chosen_centers):
                    	cnt = 0
                 	break
                if(cnt == 1):
                    break
                else:
                    continue
            
            dy = D(xs[y], centers)
    	    res = (dy * prob[x])/(dx * prob[y])

    	    uni_rand = numpy.random.uniform(0, 1, 1)
    	    
    	    if(res > uni_rand):
    	    	x = y
    	    	dx = dy
    	centers.append(densify(xs[x],l))
    	chosen_centers.append(x)
    	
    
    elapsed = timeit.default_timer() - start_time
    cluster = [None] * len(xs)
    t = 0
    start = time.time()
    count = 0
    while(count < n_iter):
        count += 1
        for i, x in enumerate(xs):
            cluster[i] = min(xrange(k), key=lambda j: dist(xs[i], centers[j]))
        for j, c in enumerate(centers):
            members = (x for i, x in enumerate(xs) if cluster[i] == j)
            centers[j] = mean(members, l)

        seeding_cost = cost(xs, centers)
        t += (time.time() - start)
    
    print("Seeding cost: %s" % (seeding_cost))
    print("Seeding time: %s seconds \n" % (elapsed))
    print("Total Running time: %s seconds " % (t))
    return cluster

with open('xs_plot.pickle','rb') as handle:
    data = pickle.load(handle)

if __name__ == '__main__':
    
    #k = [3, 5]
    k = [2]
    m = int(sys.argv[1])
    s = 0.
    with open('xs_norm.pickle', 'rb') as handle:
         xs = pickle.load(handle)

    start2 = time.time()
    for i in k:
        print("Number of clusters %d: \n" % i) 
        for p in range(3):
            print("Iteration number %d: \n" % p)
            cluster_ind = skmeanspp(i, xs, len(xs[0]))
            clusters = [set() for _ in xrange(i)]
            for q, j in enumerate(cluster_ind):
                clusters[j].add(q)
        for j, c in enumerate(clusters):
            print("cluster %d:" % j)
            print(len(c))
        plot_graph(cluster_ind,i,1000)
    print("\n \n")

    time2 = time.time() - start2
    print("Running time: %s seconds \n" % time2)
