# Clustering Algorithms for Spherical k-means

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
   
   Make sure you run `readdata.py` before running any of the SPKM algorithms.
   
3. __spherical_KMeans.py__
 
   This code clusters documents using Spherical k-means (SPKM) algorithm. It is an iterative algorithm similar to k-means (Lloyd's algorithm) except in the fact that instead of minimizing the sum of the squared distance of all the points to their nearest cluster center,  it maximizes the sum of Cosine Similarity of all the points to their nearest cluster center. Given the number of clusters, k, the first k points (cluster center candidates) are sampled uniformly at random. 
   
   To run : `python spherical_KMeans.py`
   
4. __spkm++.py__
 
   This code clusters documents using Spherical k-means++ (SPKM++) algorithm. In this algorithm, instead of initializing the k centers via uniform random sampling  (as done in SPKM), the centers are carefully chosen following an adaptive sampling strategy. The algorithm takes k passes over the dataset and samples one point in each pass. The guarantee of this sampling algorithm is that the cost of clustering obtained by considering them as cluster centers is within O(log k) factor with respect to the optimal clustering. 
   
   To run : `python spkm++.py` 
   
5. __mcmc.py__
 
   This code clusters documents using Spherical KMeans MCMC (SPKM-MC2) algorithm. This algorithm also proposes a sampling strategy and samples k initial centers. A major difference between this and SPKM++ is that it requires only one pass of the dataset to sample k points. Thus it is asymptotically/empirically faster then SPKM++ albeit offers a similar clustering guarantee.
   
   To run : `python mcmc.py` 

If you plan to use this implementation for any work of yours, please consider citing our paper,
```
@inproceedings{SPKM_MCMC,
 author    = {Rameshwar Pratap and
                   Pratheeksha Nair and
                   Anup Deshmukh and
		   Tarun Dutt},
  title     = {A Faster Sampling Algorithm for Spherical $k$-means},
  booktitle = {Asian Conference on Machine Learning (ACML), Accepted},
  year      = {2018},
}
```
