# Fast and Provable Concept Decompositions in Large Text Corpus

## Prerequisites ##

* Python 2.7 or lower; this is not Python 3 compatible.
* [NumPy](http://numpy.org)
* Libraries: [Pickle], [collections], [random], [time]

## File System 

1. __generatedata.py__

   This code outputs randomly generated real-valued data in the following format (UCI bag-of-words).
   
   ```no. of documents
   
   max. no. of words in each document (dimensionality)
   
   total number of words in vocabulary
   
   doc_id  word_id  frequency
   ```
   
   To run : `python generatedata.py > dataset/data.txt`
   
   This code can be used if you want to work with synthetically generated data. It generates data for 5 documents each with dimensionality 10. This can be changed by changing `n` and `d` within the code respectively. The frequency values are randomly chosen from between 1 and 100. 
   
2. __readdata.py__

   This code is used to generate the `xs` pickle file. It can be run either after running `generatedata.py` or for using your own dataset text file. It reads the `.txt` file and stores the sparse data in a list-of-list format and saves it as `xs.pickle`. The data is saved in the following manner : `xs[doc_id][word_id] = freq`. In order to use your own dataset, keep the `.txt` file inside `dataset` folder. 
   
   To run : `python readdata.py dataset` 
   
   Make sure you run `readdata.py` before running any of the SPKM algorithms
   
 3. __spherical_KMeans.py__
 
   This code clusters documents using Spherical KMeans (SPKM) algorithm. It is similar to KMeans (Lloyd's algorithm) except in the fact that instead of minimizing the Euclidean distance between points on a Geometric scale, it maximises their cosine similarity when they are represented on a unit sphere. 
   
   To run : `python spherical_KMeans.py`
   
 4. __spkm++.py__
 
   This code clusters documents using Spherical KMeans ++ (SPKM++) algorithm. In this algorithm, instead of initializing the k centers by sampling random points (as done in SPKM), the centers are carefully chosen from a probability distribution one at a time. So the algorithm takes k passes over the data points.
   
   To run : `python spkm++.py` 
   
 5. __mcmc.py__
 
   This code clusters documents using Spherical KMeans MCMC (SPKM-MC2) algorithm. In this algorithm, all k initial centers are chosen after a single pass over the data points from a Markov Chain sampling(whose length is set by the user). This gives the faster clustering results compared to SPKM++ and better clustering quality compared to SPKM. 
   
   To run : `python mcmc.py` 
   
