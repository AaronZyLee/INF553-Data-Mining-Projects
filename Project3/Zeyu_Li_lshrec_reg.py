import sys
from pyspark import SparkContext

if __name__ == "__main__":


    sc = SparkContext(appName="inf553")

    source = sc.textFile(sys.argv[1], 4)
    # source = sc.textFile("input.txt", 4)


    def findMin(alist):
        minVal = sys.maxint
        for i in alist:
            if i < minVal:
                minVal = i
        return minVal

    # def jac(user1Name, user2Name)

    tmp = source.map(lambda x: x.split(','))
    usersAndMovies = tmp.map(lambda x: (x[0], x[1:])).map(lambda x: (x[0], [int(i) for i in x[1]]))

    usersAndMoviesDict = {}
    for i in usersAndMovies.collect():
        usersAndMoviesDict[i[0]] = i[1]


    # the index hashing
    hashing = {}
    for itemID in range(100):
        tmpArr = range(20)
        for ind, roundNo in enumerate(tmpArr):
            tmpArr[ind] = (3 * itemID + 13 * roundNo) % 100
        hashing[itemID] = tmpArr

    #
    users = tmp.map(lambda x: x[0]).collect()
    usersAndSigs = {} # users and their signatures
    for i in users:
        usersAndSigs[i] = list()

    for i in range(0, 20):  # O(num_h)
        tmpSig = usersAndMovies.map(lambda row: (row[0], [hashing[x][i] for x in row[1]])).map(lambda row: (row[0], findMin(row[1]))).collect()  # O(num_users)
        # tmpSig = usersAndMovies.map(lambda row: (row[0], [3 * x + 13 * i for x in row[1]])).map(lambda row: (row[0], findMin(row[1]))).collect() # O(num_users)
        for j in tmpSig:
            usersAndSigs[j[0]].append(j[1])

    # print usersAndSigs.items()[:5]

    # lsh

    # temp = sc.parallelize(usersAndSigs.items(),4).map(lambda x: (x[0], [x[1][i:i+4] for i in range(0, len(x[1]), 4)])).map(lambda x: (x[0], [str(i) for i in x[1]])) # do pairwise comparison

    def func(x):
        usersAndCand = {}
        for (a,b) in x: 
            for user in b:
                if user in usersAndCand:
                    usersAndCand[user] |= set([j for j in b if j != user])
                else:
                    usersAndCand[user] = set([j for j in b if j != user])

        return usersAndCand.items()

    def top5lrg(alist): # use selection sort 5 times
        for i in range(min(5, len(alist))):
            lrgind = i
            for k in range(i + 1, len(alist)):
                if alist[k][1] > alist[lrgind][1]:
                    lrgind = k
                elif alist[k][1] == alist[lrgind][1]:
                    if int(alist[k][0][1:]) < int(alist[lrgind][0][1:]):
                        lrgind = k
            tmp = alist[i]
            alist[i] = alist[lrgind]
            alist[lrgind] = tmp

        return sorted(alist[:5], key = lambda x: int(x[0][1:]))
        
    def tagging(x):
        ans = list()
        counter = 0
        for i in x[1]:
            ans.append((str(i) + '_' + str(counter), x[0]))
            counter += 1
        return ans

    # temp = sc.parallelize(usersAndSigs.items(),4).map(lambda x: (x[0], [x[1][i:i+4] for i in range(0, len(x[1]), 4)])).flatMap(lambda x: [(str(i), x[0]) for i in x[1]]).groupByKey().map(lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) > 1).map(func).groupByKey()
    # temp = sc.parallelize(usersAndSigs.items(),4).map(lambda x: (x[0], [x[1][i:i+4] for i in range(0, len(x[1]), 4)])).flatMap(lambda x: [(str(i), x[0]) for i in x[1]]).groupByKey().map(lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) > 1).flatMap(lambda x: [(i, [j for j in x[1] if j != i]) for i in x[1]]).groupByKey()

    # make the section correspond
    temp = sc.parallelize(usersAndSigs.items(),4).map(lambda x: (x[0], [x[1][i:i+4] for i in range(0, len(x[1]), 4)]))
    # print temp.collect()[:2]
    # here need to creat a func to make sure corresponds
    temp2 = temp.flatMap(tagging).groupByKey().map(lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) > 1).mapPartitions(func).reduceByKey(lambda x,y: list(set(x) | set(y)))
    temp3 = temp2.map(lambda x: (x[0], [(i, float(len(set(usersAndMoviesDict[x[0]]) & set(usersAndMoviesDict[i]))) / float(len(set(usersAndMoviesDict[x[0]]) | set(usersAndMoviesDict[i])))) for i in x[1]])).mapValues(top5lrg).map(lambda x: (x[0], [i[0] for i in x[1]]))

    result = temp3.collect()
    result = sorted(result, key = lambda x: int(x[0][1:]))

    # print temp3.collect()[:3]


    # text_file = open("outputcorrect.txt", "w")
    text_file = open(sys.argv[2], "w")
    
    for i in result:
        text_file.write(str(i[0]) + ":")
        for j,k in enumerate(i[1]):
            if j == len(i[1]) - 1:
                text_file.write(str(k))
            else:
                text_file.write(str(k) + ",")
        text_file.write("\n")














