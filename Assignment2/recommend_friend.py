"""
Friend recommendation by mutual-frind count,
using Spark RDD MapReduce """

import os
import sys
import urllib.request
from operator import add

from pyspark.sql import SparkSession

URL = "https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt"
LOCAL = "soc-LiveJournal1Adj.txt"
SAMPLE_SIZE = 10
TOP_K = 10

def parse(line):
    parts = line.split("\t")
    if len(parts) < 2 or not parts[0].strip():
        return None
    friends = [int(x) for x in parts[1].split(",") if x.strip()]
    return (int(parts[0]), friends)

def both_directions(rec):
    """Frindship is undirected: emit (u,f) for every listed edge."""

    u, fs = rec
    for f in fs:
        yield (u, f)
        yield (f, u)

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if not os.path.exists(LOCAL):
        urllib.request.urlretrieve(URL, LOCAL)

    spark = SparkSession.builder.appName("FriendRecommender").getOrCreate()

    sc = spark.sparkContext

    #Build symmetric adjacency lists
    raw = sc.textFile("file:///" + os.path.abspath(LOCAL))
    adj = (raw.map(parse)
           .filter(lambda r: r is not None)
           .flatMap(both_directions)
           .distinct()
           .groupByKey()
           .mapValues(lambda it: sorted(set(it)))
           .cache())

    #Random subset of 10 users (only those users that have friends)
    sample = adj.keys().takeSample(False, SAMPLE_SIZE, seed)
    sample_set = set(sample)
    direct = adj.filter(lambda kv: kv[0] in sample_set).collectAsMap()
    b_sample = sc.broadcast(sample_set)
    b_direct = sc.broadcast({u: set(fs) for u, fs in direct.items()})

    #MAP: for each user m with friend list L, m is a mutual friend
    #of every pair (s, c) with s, c in L. Only emit pairs whose
    #first element is sampled user, and skip existing friends.
    def emit(kv):
        _, friends = kv
        targets = [s for s in friends if s in b_sample.value]
        for s in targets:
            already = b_direct.value[s]
            for c in friends:
                if c != s and c not in already:
                    yield ((s, c), 1)

    #REDUCE: SUM 1s -> number of mutual friends per (s, c)
    counts = adj.flatMap(emit).reduceByKey(add)

    #Regroup by user, rank: count desc, then userID asc, keep top K
    def top_k(it):
        ranked = sorted(it, key=lambda cn: (-cn[1], cn[0]))
        return [c for c, _ in ranked[:TOP_K]]

    recs = (counts.map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
            .groupByKey()
            .mapValues(top_k)
            .collectAsMap())

    #Display (users with zero candidates print an empty list)
    for u in sample:
        print("%d\t%s" % (u, ",".join(str(x) for x in recs.get(u, []))))

    spark.stop()



if __name__ == "__main__":
    main()