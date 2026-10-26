# Part 3: Naive Bayes with Spark MapReduce

## 1. Dataset

20 Newsgroups (by-date version): about 18,800 posts in 20 topic classes (`alt.atheism`,
`comp.graphics`, ..., `talk.religion.misc`).

- Public location: `http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz` (original
  distribution). **TODO before submitting:** upload a copy to your own public location
  (S3 / GitHub) and set `DEFAULT_URL` in `naive_bayes.py` to it.
- The script downloads the archive into the working directory and extracts it there (no absolute paths).

## 2. Preprocessing (PySpark)

1. `wholeTextFiles` reads every post as `(path, text)`; the class is the parent folder name.
2. Header removal (everything before the first blank line) and removal of quoted lines (`>`).
   Headers contain the newsgroup name and would leak the label.
3. Lowercase and tokenize with the regex `[a-z]{3,}`.
4. Stop-word removal with Spark's `StopWordsRemover` default English list (used only as a word list).
5. Stemming with a small suffix-stripping stemmer (`sses`, `ies`, `ing`, `ed`, `ly`, plural `s`),
   implemented in the script so there is no extra dependency.
6. Documents with no tokens left are dropped.

## 3. Split

The original train and test folders are pooled, then split randomly 80/20 with `randomSplit(seed=42)`.

## 4. Model

Multinomial Naive Bayes with Laplace smoothing (alpha = 1), computed in log space.

- Prior: `P(c) = N_c / N`
- Likelihood: `P(w | c) = (count(w, c) + alpha) / (T_c + alpha * |V|)`, where `T_c` is the total number of
  vocabulary-word occurrences in class `c` and `V` is the vocabulary (words with training count >= 3).
- Prediction: `argmax_c [ log P(c) + sum_w tf(w) * log P(w | c) ]`; out-of-vocabulary words are ignored.

## 5. MapReduce pseudocode

```
# Job A - class priors
map    (label, tokens)                 -> emit (label, 1)
reduce (label, [1,1,...])              -> (label, N_label)

# Job B - term frequencies per class
map    (label, tokens)                 -> for each (word, tf) in Counter(tokens):
                                              emit ((label, word), tf)           # in-mapper combining
reduce ((label, word), [tf,...])       -> ((label, word), sum)

# Regroup (second shuffle)
map    ((label, word), n)              -> emit (word, (classIndex(label), n))
combine/reduce by word                 -> (word, [n_class0, ..., n_classK-1])
filter sum(vector) >= MIN_COUNT        -> vocabulary V

# Driver (K x |V| values, small)
T_c            = column sums of the count matrix
log P(c)       = log(N_c / N)
log P(w | c)   = log((n_wc + alpha) / (T_c + alpha * |V|))
broadcast(model)

# Testing - map-only job
map (label, tokens):
    scores = log P(c) for all c
    for (w, tf) in Counter(tokens): scores[c] += tf * log P(w | c)  (w in V)
    emit (label, argmax_c scores)
# driver: accuracy, per-class precision / recall / F1
```

Cost: training reads the corpus once per job and shuffles `O(distinct (class, word) pairs)` records.
Testing is embarrassingly parallel.

## 6. Results

=== Naive Bayes (multinomial, from scratch on Spark RDDs) ===
seed=42  train docs=15011  test docs=3799  vocab=31903  alpha=1.00  min_count=3

class                            docs      prior
alt.atheism                       613    0.04084
comp.graphics                     782    0.05210
comp.os.ms-windows.misc           783    0.05216
comp.sys.ibm.pc.hardware          783    0.05216
comp.sys.mac.hardware             766    0.05103
comp.windows.x                    795    0.05296
misc.forsale                      788    0.05249
rec.autos                         799    0.05323
rec.motorcycles                   781    0.05203
rec.sport.baseball                774    0.05156
rec.sport.hockey                  787    0.05243
sci.crypt                         782    0.05210
sci.electronics                   815    0.05429
sci.med                           792    0.05276
sci.space                         804    0.05356
soc.religion.christian            769    0.05123
talk.politics.guns                731    0.04870
talk.politics.mideast             737    0.04910
talk.politics.misc                631    0.04204
talk.religion.misc                499    0.03324

class                        precision    recall        f1  support
alt.atheism                     0.8020    0.8541    0.8272      185
comp.graphics                   0.6794    0.7553    0.7154      188
comp.os.ms-windows.misc         0.9151    0.4924    0.6403      197
comp.sys.ibm.pc.hardware        0.6379    0.7908    0.7062      196
comp.sys.mac.hardware           0.7794    0.8238    0.8010      193
comp.windows.x                  0.7560    0.8272    0.7900      191
misc.forsale                    0.8671    0.6966    0.7726      178
rec.autos                       0.8657    0.9206    0.8923      189
rec.motorcycles                 0.8755    0.9533    0.9128      214
rec.sport.baseball              0.9171    0.9087    0.9128      219
rec.sport.hockey                0.9336    0.9336    0.9336      211
sci.crypt                       0.9126    0.8995    0.9060      209
sci.electronics                 0.8000    0.7574    0.7781      169
sci.med                         0.8794    0.8929    0.8861      196
sci.space                       0.9261    0.8956    0.9106      182
soc.religion.christian          0.8207    0.9035    0.8601      228
talk.politics.guns              0.7700    0.9162    0.8367      179
talk.politics.mideast           0.9167    0.8670    0.8911      203
talk.politics.misc              0.7419    0.8042    0.7718      143
talk.religion.misc              0.8649    0.4961    0.6305      129

ACCURACY = 0.8281  (3146 / 3799)
Macro-F1 = 0.8188

Observations to write up from your run: which classes are confused most (for example `comp.*`
groups with each other, or `talk.religion.misc` with `alt.atheism`), and how accuracy compares
with the majority-class rate (about 5%, since there are 20 roughly balanced classes).

## 7. Notes and limitations

- Naive Bayes assumes words are conditionally independent given the class, which is false for text, but it
  still works well as a baseline.
- `MIN_COUNT = 3` and `alpha = 1` are fixed defaults, not tuned.
- The stemmer is deliberately light and will not match Porter or Snowball exactly.

## 8. How to run

```
spark-submit naive_bayes.py 42
```