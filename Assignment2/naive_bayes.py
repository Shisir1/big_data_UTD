"""
Multinomial Naive Bayes implemented from scratch with Spark RDD MapReduce.
Dataset: 20 Newsgroups (by-date version), 20 classes, ~18.8k documents.

Usage:  spark-submit naive_bayes.py [seed] [dataset_url]
"""
import math
import os
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from operator import add
from pathlib import Path

from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SparkSession

DEFAULT_URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
ARCHIVE = "20news-bydate.tar.gz"
DATA_DIR = "data_20news"
RESULTS_FILE = "nb_results.txt"

TRAIN_FRAC = 0.8
ALPHA = 1.0       # Laplace smoothing
MIN_COUNT = 3     # drop words seen fewer than this many times in the training set

# Data download / extraction (plain Python, no model logic)
def prepare_data(url):
    if not os.path.exists(ARCHIVE):
        print("Downloading %s ..." % url)
        urllib.request.urlretrieve(url, ARCHIVE)
    if not os.path.isdir(DATA_DIR):
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(DATA_DIR, filter="data")
            else:
                tf.extractall(DATA_DIR)
    for root, dirs, _ in os.walk(DATA_DIR):
        if "20news-bydate-train" in dirs:
            return root
    raise RuntimeError("Could not find 20news-bydate-train inside %s" % DATA_DIR)

# Preprocessing: header/quote removal, tokenization, stop words, stemming
def light_stem(w):
    """Small suffix-stripping stemmer (Porter step-1 style, no external library)."""
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("ly") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def make_preprocessor(stop_bc):
    def process(kv):
        path, text = kv
        label = path.rstrip("/").split("/")[-2]
        # drop the message header (everything before the first blank line);
        # headers leak the newsgroup name and would inflate accuracy
        body = text.split("\n\n", 1)[1] if "\n\n" in text else text
        # drop quoted lines from earlier posts
        body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith(">"))
        stop = stop_bc.value
        tokens = [light_stem(w) for w in re.findall(r"[a-z]{3,}", body.lower())
                  if w not in stop]
        return (label, tokens)
    return process


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_URL

    root = prepare_data(url)

    spark = SparkSession.builder.appName("NaiveBayesMapReduce").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # 1. Preprocess with PySpark
    stop_bc = sc.broadcast(set(StopWordsRemover.loadDefaultStopWords("english")))
    #pattern = "file://" + os.path.abspath(root) + "/20news-bydate*/*/*"
    # Resolve the dataset directory relative to this Python script
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data_20news"

    # Match the dataset's category and document directories
    pattern = (data_dir / "20news-bydate-*" / "*" / "*").as_posix()
    docs = (sc.wholeTextFiles(pattern, minPartitions=8)
              .map(make_preprocessor(stop_bc))
              .filter(lambda d: len(d[1]) > 0)
              .cache())

    # 2. Random train/test split (original train+test pooled, then re-split)
    train, test = docs.randomSplit([TRAIN_FRAC, 1.0 - TRAIN_FRAC], seed)
    train.cache()
    test.cache()

    # 3. Training with MapReduce
    #    Job A: class document counts          -> priors
    #    Job B: (class, word) term frequencies -> likelihoods
    # Job A  map: doc -> (class, 1)   reduce: sum
    class_docs = train.map(lambda d: (d[0], 1)).reduceByKey(add).collectAsMap()
    classes = sorted(class_docs)
    K = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    n_train = sum(class_docs.values())
    log_prior = [math.log(class_docs[c] / n_train) for c in classes]

    # Job B  map: doc -> ((class, word), tf)  (in-mapper combining via Counter)
    #        reduce: sum
    word_class_counts = (train
                         .flatMap(lambda d: (((d[0], w), n) for w, n in Counter(d[1]).items()))
                         .reduceByKey(add))

    # Regroup by word into a K-vector of counts, drop rare words
    def create(t):
        v = [0] * K
        v[t[0]] += t[1]
        return v

    def merge_value(v, t):
        v[t[0]] += t[1]
        return v

    def merge_comb(a, b):
        return [x + y for x, y in zip(a, b)]

    word_vectors = (word_class_counts
                    .map(lambda kv: (kv[0][1], (cidx[kv[0][0]], kv[1])))
                    .combineByKey(create, merge_value, merge_comb)
                    .filter(lambda kv: sum(kv[1]) >= MIN_COUNT)
                    .collectAsMap())

    V = len(word_vectors)
    total_tokens = [0] * K  # total word occurrences per class (restricted to vocabulary)
    for vec in word_vectors.values():
        for i in range(K):
            total_tokens[i] += vec[i]
    denom = [total_tokens[i] + ALPHA * V for i in range(K)]
    # log P(word | class) with Laplace smoothing
    log_likelihood = {w: [math.log((vec[i] + ALPHA) / denom[i]) for i in range(K)]
                      for w, vec in word_vectors.items()}
    del word_vectors

    model_bc = sc.broadcast((classes, log_prior, log_likelihood))

    # 4. Testing: map each test doc to its argmax class
    def predict(d):
        cls, prior, loglik = model_bc.value
        scores = list(prior)
        for w, n in Counter(d[1]).items():
            lp = loglik.get(w)
            if lp is not None:
                for i in range(len(scores)):
                    scores[i] += n * lp[i]
        best = max(range(len(scores)), key=scores.__getitem__)
        return (d[0], cls[best])

    preds = test.map(predict).collect()

    # 5. Evaluation (plain Python metrics) and report
    n_test = len(preds)
    correct = sum(1 for t, p in preds if t == p)
    true_c, pred_c, tp = Counter(), Counter(), Counter()
    for t, p in preds:
        true_c[t] += 1
        pred_c[p] += 1
        if t == p:
            tp[t] += 1

    out = []
    out.append("=== Naive Bayes (multinomial, from scratch on Spark RDDs) ===")
    out.append("seed=%d  train docs=%d  test docs=%d  vocab=%d  alpha=%.2f  min_count=%d"
               % (seed, n_train, n_test, V, ALPHA, MIN_COUNT))
    out.append("")
    out.append("%-28s %8s %10s" % ("class", "docs", "prior"))
    for i, c in enumerate(classes):
        out.append("%-28s %8d %10.5f" % (c, class_docs[c], math.exp(log_prior[i])))
    out.append("")
    out.append("%-28s %9s %9s %9s %8s" % ("class", "precision", "recall", "f1", "support"))
    f1s = []
    for c in classes:
        p = tp[c] / pred_c[c] if pred_c[c] else 0.0
        r = tp[c] / true_c[c] if true_c[c] else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f)
        out.append("%-28s %9.4f %9.4f %9.4f %8d" % (c, p, r, f, true_c[c]))
    out.append("")
    out.append("ACCURACY = %.4f  (%d / %d)" % (correct / n_test, correct, n_test))
    out.append("Macro-F1 = %.4f" % (sum(f1s) / len(f1s)))

    text = "\n".join(out)
    print(text)
    with open(RESULTS_FILE, "w") as fh:
        fh.write(text + "\n")

    spark.stop()


if __name__ == "__main__":
    main()