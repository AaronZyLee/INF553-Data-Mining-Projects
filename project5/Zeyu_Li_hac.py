from __future__ import division
import sys
import heapq
import numpy as np
from scipy.sparse import csc_matrix


# cosine similarity
def cos_sim(a, b):  # a and b are two csc represented array
    return float(a.multiply(b).sum()) / (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))

if __name__ == "__main__":
    ## transfer each document into a tf-idf vector

    # line 1: number of documents in the collection (num_docs)
    # line 2: number of words in the vocabulary (vocabulary only contains words appear in at least 10 docs) (num_vocab)
    # line 3: number of words that appear in at least 1 document (num_words)
    # others: <doc id> <word id> <tf>
    # word id ranges from 1 - num_vocab, so there are num_vocab slots in uncompressed vector, each slot is tfidf

    desired_num_clusters = int(sys.argv[2])
    # desired_num_clusters = 3

    # "case_s/docword.enron_s.txt"
    # "case/docword.enron_s.txt"
    with open(sys.argv[1]) as f:
       lines = f.readlines()
    # with open("case_s/docword.enron_s.txt") as f:
    #   lines = f.readlines()

    num_docs = int(lines[0])
    num_vocab = int(lines[1])
    num_words = int(lines[2])

    # tf
    maxIndex = 0
    docAndWordFreq = {} # {1: [(1,1),(2,1),(5,1),...], 2: [],...}
    df = {} # key: int, value: int
    for i in lines[3:]:
        tmp = i.split(' ') # tmp is (1, 118, 1)
        # prepare word and frequency
        if int(tmp[0]) in docAndWordFreq:
            docAndWordFreq[int(tmp[0])].append((int(tmp[1]), int(tmp[2])))
        else:
            docAndWordFreq[int(tmp[0])] = [(int(tmp[1]), int(tmp[2]))]

        # prepare idf
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

    # tfidf: {docID:tfidfVector, ...}
    tfidf = {}
    for i in docAndWordFreq:
        col = []
        data = []
        for j in docAndWordFreq[i]: # j is like (1,3)
            col.append(j[0])
            data.append(j[1] * idf[j[0]]) # tf * idf

        row = np.array([0 for x in range(len(col))])
        col = np.array(col)

        norm_factor = float(np.linalg.norm(data))

        # unnormed = csc_matrix((data, (row, col)), shape=(1, total_num_of_words))
        # tfidf_un[i] = csc_matrix((data, (row, col)), shape=(1, total_num_of_words)).toarray()  #.toarray() # to see how array look like
        data_normed = [ele / norm_factor for ele in data]
        tfidf[i] = csc_matrix((data_normed, (row, col)), shape=(1, maxIndex + 1)) ######

    # print tfidf.items() ######
    # exit()  #######
    ## finish constructing tfidf vector for all docs

    # initialize clusterAndSummary
    # the old cluster ID was 1,2,3...,num_docs for single doc cluster
    # new clusterID will be given to newly formed clusters
    clusterAndSummary = {} # {clusterID:([sumValues]', numPoints, [centroid]', [points]), ...}
    for i in tfidf:
        clusterAndSummary[i] = (tfidf[i], 1, tfidf[i], [i])

    clusterID = num_docs + 1

    # initialize clusterAndPoints
    clusterAndPoints = {} # {clusterID:[doc1, doc2, ...], ...}
    for i in tfidf:
        clusterAndPoints[i] = [i]

    # compute pairwise distance of all initial clusters
    # pairWiseDist = [(distance, clusterID1, clusterID2), ...]
    tfidfList = tfidf.items() # [(id1, [tfidf_vec]), ...]
    pairWiseDist = []
    for i,j in enumerate(tfidfList): # i is index, j is actual item
        for k in tfidfList[i + 1:]:
            pairWiseDist.append((-cos_sim(j[1], k[1]), j[0], k[0]))


    # build priority queue
    # pairWiseDist: [(neg_cos, id1, id2)]
    heapq.heapify(pairWiseDist)

    # print [heapq.heappop(pairWiseDist) for i in range(len(pairWiseDist))] # this can be printed in a sorted order
    while len(clusterAndSummary) > desired_num_clusters:
        # pop the top of heap and get two nearest clusters
        pair = heapq.heappop(pairWiseDist) # (neg_cos, id1, id2)
        id1 = pair[1]
        id2 = pair[2]
        if not((id1 in clusterAndSummary) and (id2 in clusterAndSummary)): continue

        # merge two clusters and generate new summary
        # clusterAndSummary: {clusterID:([sumValues]', numPoints, [centroid]', [points]), ...}
        summary_cluster_1 = clusterAndSummary[id1]
        summary_cluster_2 = clusterAndSummary[id2]

        del clusterAndSummary[id1]
        del clusterAndSummary[id2]

        new_sumValues = summary_cluster_1[0] + summary_cluster_2[0]
        new_numPoints = summary_cluster_1[1] + summary_cluster_2[1]
        new_centroid = new_sumValues / float(new_numPoints)
        new_points = summary_cluster_1[3] + summary_cluster_2[3]

        # update pairwise distance heap
        for i in clusterAndSummary: # i is cluster ID
            heapq.heappush(pairWiseDist, (-cos_sim(new_centroid, clusterAndSummary[i][2]), i, clusterID))

        clusterAndSummary[clusterID] = (new_sumValues, new_numPoints, new_centroid, new_points)

        clusterID += 1

    for i,j in clusterAndSummary.items():
        print ",".join(str(x) for x in j[3])






    # tst = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8]}
    # pairWiseDis = []
    # tstList = tst.items()
    # for i,j in enumerate(tstList):
    #     for k in tstList[i + 1:]:
    #         pairWiseDis.append((j[1] + k[1], j[0], k[0]))


    # tups = []
    # for i in lines[3:]:
    #     tmp = i.split(' ')
    #     tups.append((int(tmp[0]), int(tmp[1]), int(tmp[2])))
    # import operator
    # print sorted(tups,key=operator.itemgetter(0))[:50]
    # exit()








