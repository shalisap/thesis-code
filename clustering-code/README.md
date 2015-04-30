thesis-code
===========

Honors Senior Thesis Code

The Effects of Clustering Technique on Simulation of the Tor Network

Tor is a low-latency network built for anonymity online. Due to ethical concerns, attacks and protocol changes cannot be attempted on the actual network and as a result, considerable work has been done in an attempt to precisely model the network itself. While many simulation platforms attempt to model the Tor network for research purposes, most platforms contain simple synthetic models of the behavior of actual Tor users. Without an appropriate model of network traffic, the simulated network cannot fully mimic the real Tor network, potentially weakening any previous conclusions. This thesis presents an examination of the effects of clustering algorithm and distance measure choices on the quality of a user model. In particular, the goal is to improve a previously established generative model of outbound data with the greater goal of realistically simulating user traffic of the Tor network.

The clustering algorithms (k-Means, k-Medoids, hierarchical agglomerative clustering and associated agglomeration methods), distance functions (Euclidean, Manhattan, Edit Distances), and evaluation code (Rand index, adjusted Rand index, Collapsed Pairs) are implemented with Java 8. 

In addition, in Python 2.7.3, modifications were made to \citet{Julian} code to allow for preprocessing of data collected from Shadow, visualization, and series modeling of pairs of inbound, outbound cell counts as well as for the clustering evaluation. 


Requirements:
-----------------
Java:
* Weka 3.6.11+
* Jahmm 0.6.1

For the driver class clustering/runClustering:
* JSON.simple 1.1.1
* Gson 2.3.1

For testing Java code:
* JUnit 4.12
* Hamcrest-core 1.3
* Mockito 1.10.14

Python:
Available on Pip:
* scikit-learn 0.13.1+
* scipy 0.10+
* Pycluster
* fastcluster

Manual install:
* ghmm (http://ghmm.org)
