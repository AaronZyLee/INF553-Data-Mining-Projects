import sys
from pyspark import SparkContext

if __name__ == "__main__":
    if (len (sys.argv)) != 4:
        print "Usage: Haotian_Zhang_TwoPhase.py mat-A/values.txt mat-B/values.txt output.txt"
        exit(1)
        
    sc = SparkContext(appName = "inf553")
    
    def myFunc(x):
        APart = []
        BPart = []
        outcome = []
        for i in x[1]:
            if i[0] == "A":
                APart.append(i)
            else:
                BPart.append(i)
        for i in APart:
            for j in BPart:
                outcome.append(((i[1],j[1]), i[2] * j[2]))
        return outcome
    
    matA = sc.textFile(sys.argv[1]) # RDD matA and matB are now collections of lines 
    matB = sc.textFile(sys.argv[2])
    
    # matA = sc.textFile('mat-A/values.txt')
    # matB = sc.textFile('mat-B/values.txt')
    
    MAMapped = matA.map(lambda line: line.split(',')).map(lambda x: (x[1], ('A', x[0], int(x[2]))))
    MBMapped = matB.map(lambda line: line.split(',')).map(lambda x: (x[0], ('B', x[1], int(x[2]))))
    
    outcome =  MAMapped.union(MBMapped).groupByKey().map(lambda tup: (tup[0], [x for x in tup[1]])).flatMap(myFunc).reduceByKey(lambda a,b:a+b).collect()
    text_file = open(sys.argv[3], "w")
    for i in outcome:
        text_file.write(i[0][0] + "," + i[0][1] + "\t" + str(i[1]) + "\n")
