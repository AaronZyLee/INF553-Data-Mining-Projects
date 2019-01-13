#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function
import math
import sys

import numpy as np
from scipy.sparse import csc_matrix
from pyspark.sql import SparkSession

# cosine similarity of two csc represented array
def cos_sim(a, b):
    return float(a.multiply(b).sum()) / (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))

# def cos_sim(a, b):
#   return float(sum(a * b)) / (np.linalg.norm(a) * np.linalg.norm(b))

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

# usage: closestPoint(p, kPoints)
def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = cos_sim(p, centers[i])
        # tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("PythonKMeans") \
        .getOrCreate()

    ##
    with open("case_s/docword.enron_s.txt") as f:
        linesTmp = f.readlines()

    num_docs = int(linesTmp[0])
    num_vocab = int(linesTmp[1])
    num_words = int(linesTmp[2])

    maxIndex = 0
    df = {}
    for i in linesTmp[3:]:
        tmp = i.split(' ')  # tmp is (1, 118, 1)

        if int(tmp[1]) in df:
            df[int(tmp[1])] += 1
        else:
            df[int(tmp[1])] = 1

        if int(tmp[1]) > maxIndex:
            maxIndex = int(tmp[1])

    # formalize into idf
    idf = {}
    for i in df:
        idf[i] = np.log2((num_docs + 1.0) / float(df[i] + 1.0))

    f.close()
    linesTmp = None
    ##

    ## process doc data
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    # lines = spark.read.text("case_s/docword.enron_s.txt").rdd.map(lambda r: r[0])
    data = lines.map(parseVector).filter(lambda x: len(x) > 1).cache()

    K = int(sys.argv[2])
    # K = 3
    convergeDist = float(sys.argv[3])
    # convergeDist = 0.00001
    outputFileName = sys.argv[4]
    # outputFileName = "out1.txt"

    ##
    data = data.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(x[1]))).map(
        lambda x: (x[0], ([i[0] for i in x[1]], [i[1] * idf[i[0]] for i in x[1]]))).map(
        lambda x: (x[0], (x[1][0], [i / np.linalg.norm(x[1][1]) for i in x[1][1]]))).map(
        lambda x: (x[0], csc_matrix((np.array(x[1][1]), (np.array([0 for i in range(len(x[1][0]))]), np.array(x[1][0]))),
                                    shape=(1, maxIndex + 1)))).sortByKey().map(lambda x: x[1])

    # print (data.collect())


    kPoints = data.repartition(1).takeSample(False, K, 1)

    tempDist = 1.0

    while tempDist > convergeDist:

        closest = data.map(lambda p: (closestPoint(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
        tempDist = sum(np.linalg.norm((kPoints[iK] - p).toarray()) for (iK, p) in newPoints)
        print (tempDist)

        for (iK, p) in newPoints:
            kPoints[iK] = p

    print("Final centers: " + str(kPoints))

    spark.stop()

    with open(outputFileName, "w") as f:
        for i in kPoints:
            f.write('{}\n'.format(str(len(i.nonzero()[1]))))
