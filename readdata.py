import numpy as np
from collections import defaultdict
from math import sqrt
import random
import pickle
from numpy import linalg as la

def preprocessing(xs):
# adding 0 frequency for all the words that do not occur in a particular doc
    for x in xs:
        for i in range(len(x)):
            if(x[i] == 0):
                 #x[i] = tuple((i,0.))
                 x[i] = 0.
    return xs


def normalise(xs):
# This function normalises the points in the document corupus so that they can be represented on a unit sphere    
    norms = la.norm(xs, axis = 1)
    xs = xs/norms[:,None]        
    return xs  

if __name__ == '__main__':
    # Cluster a bunch of text documents.
    import re
    import sys
    import os
    import glob
    
  
    s = 0.
    args = sys.argv[1]
    path = os.getcwd() + "/" + args + "/*.txt"
    
    file_sys = []
    # Reading input data in 'doc' 'word' 'freq' format
    for filename in glob.glob(path):
        #x = defaultdict(float)
        rev = filename[::-1]
        d = rev.index('/')
        name = rev[0:d][::-1]
        file_sys.append(name)
        
        with open(filename) as f:
            t = f.readlines()
            f.seek(0)
            n = t[0][0:len(t[0])-1]
            s = t[1][0:len(t[1])-1]
            w = t[2][0:len(t[2])-1]
            go_out = 0   			
            xs = [[] for i in range(int(n))]
            #xs = [[] for i in range(int(4))]
            print n
            print s	
            x = [0.] * int(s)
            
            l = t[3]
            data = l.split();
            prev_doc = int(data[0]) - 1;
            curr_doc = prev_doc
            
            for line in t[3:]:
                l = line[0:len(line)-1]
                data = l.split()
                curr_doc = int(data[0])-1
                word = int(data[1])-1
                freq = int(data[2])
                
               	if(curr_doc == prev_doc):
                	x[word] = freq
  
                else:

                    prev_doc = curr_doc
                    x = [0.] * int(s)
                    x[word] = freq

                xs[curr_doc] = x
                
            xs = preprocessing(xs)

    print xs
    xs = normalise(xs)
    print("done")
    
    xs = np.array(xs)
    
    with open('xs.pickle', 'wb') as handle:
         pickle.dump(xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
