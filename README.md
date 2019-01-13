# Data-Mining-Projects-INF553-Python-Java #
Programming Projects for Course Data Mining INF553, in Python and Java.  
These projects are run on Linux EC2 instances built on AWS. Here are short descriptions. Pls check specific folder for detail.  
## Project1: Matrix Muliplication via MapReduce: realized by using Hadoop in Java and Spark in Python on AWS. ##
It contains  
1) A Hadoop MapReduce program, TwoPhase.java, that computes the multiplication of two large-scale matrices using the two-phase approach (https://adhoop.wordpress.com/2012/03/28/matrix_multiplication_2_step/).  
2) A Spark MapReduce program, TwoPhase.py, that implements the same 2-phase approach in Spark.  
For more info and execution instructions, please see hw1.pdf insider the folder.  
## Project2: Finding Frequent Itemsets over millions shopping baskets: realized by SON algorithm on Spark in Python on AWS. ##
This project aims to discovered correlated items customers most likely to purchase together given big shopping basket dataset. It is implemented by SON Data Mining Algorithm via MapReduce with Spark on AWS, programming in Python. It calculates frequent item sets over millions of shopping basket records in just a few (<30) seconds.  
For more info and execution instructions, please see hw2.pdf insider the folder.  
## Project3: Finding similar users over millions movie watching history: realized by Local Sensitive Hashing (LSH) algorithm on Spark in Python on AWS. ##
This project aims to find similar users over millions movie watching history. It is implemented using LSH algorithm.  
For more info and execution instructions, please see hw3.pdf insider the folder.  
## Project4: Recommendation System: UV Decomposition and ALS Algorithm in Collaborative Filtering, realized by Python and Spark on AWS. ##
This project performs the latent factor modeling of utility matrix M. In latent factor modeling, the goal is to decompose M into lower-rank matrices U and V such that the difference between M and UV is minimized. It is a critical step in building a Recommendation System.  
For more info and execution instructions, please see hw4.pdf insider the folder.  
## Project5: Document clustering: implemented in 2 ways: Hierarchical clustering in Python, and Spherical K-means clustering in Spark on AWS. ##
This project contains two document clustering algorithms. The first is hierarchical (agglomerative/bottom-up) clustering HAC (in Python), and the second is (spherical) k-means (in Spark). K-means on unit vectors is often called spherical k-means.  
For more info and execution instructions, please see hw5.pdf insider the folder.  