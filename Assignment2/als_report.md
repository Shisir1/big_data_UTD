# Part 2: Friend Recommendation with ALS

## 1. Data preprocessing

1. **Parse and symmetrize.** Each line `uid<TAB>f1,f2,...` becomes edges `(uid, f)`. Every edge is
   also added reversed and the set is deduplicated, because friendship is undirected.
2. **Same 10 users as Part 1.** Both scripts take the sorted list of user IDs and draw
   `random.Random(seed).sample(ids, 10)` with seed 42, so the subset is identical and does not
   depend on Spark partitioning.
3. **Candidate universe (scale-down).** Training ALS plus a cross-validated grid on all ~4.8M users
   is impractical on a single machine, so the model is trained on the induced subgraph of the
   sampled users, their friends, and their friends-of-friends (2 hops).
   - Every mutual-friend candidate of a sampled user lies inside this set, and every path
     `s - m - c` is preserved, so mutual-friend counts inside the subgraph equal the full-graph
     counts for the sampled users.
4. **Ratings `(user, item, rating)`.** Each user is also an item.
   - Existing friendship: `rating = 1.0` (both directions).
   - **Negative sampling:** for each positive row, one random node from the universe is drawn as
     an item with `rating = 0.0` (rows that collide with a real friendship or the user
     themselves are removed).
   - Reason: with only 1.0 ratings, RMSE is trivially minimized by predicting 1 for everything and
     cannot rank hyperparameters meaningfully.
5. **Split.** 80% train / 20% test (seeded). Cross-validation runs on the train part only; the
   test part is used once for the final RMSE.

## 2. Modeling approach

- **Pipeline** = `[StringIndexerModel(user -> userIdx), StringIndexerModel(item -> itemIdx), ALS]`.
  - Both indexers are built with `StringIndexerModel.from_labels` from one shared vocabulary
    (all nodes), so index *k* means the same person in the user and item columns. Fitting two
    ordinary `StringIndexer`s (especially inside CV folds) can yield different mappings per column,
    which would silently corrupt the factor model.
- **ALS:** explicit feedback on the 0/1 ratings, `coldStartStrategy="drop"` (so CV metrics are not NaN),
  `nonnegative=False`.
- **ParamGridBuilder:** `rank` in {10, 30}, `regParam` in {0.05, 0.2}, `maxIter` in {5, 10}: 8 combinations.
- **CrossValidator:** 3 folds, metric **RMSE** (`RegressionEvaluator`). The best pipeline is
  refit on the full train split.
- A constant-mean predictor is evaluated on the test split as a sanity baseline.

## 3. Recommendation generation

`ALSModel.recommendForUserSubset` is asked for `10 + max_degree + 2` items per sampled user.
Self and existing friends are removed, then the top 10 by predicted score are kept. Output format:
`UserID<TAB>id1,id2,...` (written to `als_recommendations.tsv` and printed).

## 4. Comparison with the MapReduce method

The script prints, per user, the overlap@10 between both lists and the average number of mutual
friends of each list.

**Expected differences (reasoning, not measured results):**

- *Objective.* MapReduce ranks by exact mutual-friend count, so it is optimal for that criterion
  by construction. ALS optimizes reconstruction of the 0/1 matrix, so mutual-friend count is only an
  indirect signal.
- *Popularity effect.* Latent factors tend to favor high-degree users, which can displace
  candidates with more mutual friends but lower overall connectivity.
- *Reach.* MapReduce can only suggest users with at least one mutual friend. ALS can score any
  node in the universe, so some suggestions may have zero mutual friends.
- *Determinism and cost.* MapReduce is exact, deterministic, and explainable ("you share 14
  friends"). ALS depends on negative sampling and the random initialization, needs tuning, and
  is much more expensive, but it can generalize to users whose neighborhoods overlap weakly.

**Observed results:**

| Metric | Value |
|---|---|
=== Cross-Validation RMSE per setting ===
 {'rank': 10, 'regParam': 0.05, 'maxIter': 5} -> 0.4288
 {'rank': 10, 'regParam': 0.05, 'maxIter': 10} -> 0.3947
 {'rank': 10, 'regParam': 0.2, 'maxIter': 5} -> 0.4359
 {'rank': 10, 'regParam': 0.2, 'maxIter': 10} -> 0.4239
 {'rank': 30, 'regParam': 0.05, 'maxIter': 5} -> 0.3848
 {'rank': 30, 'regParam': 0.05, 'maxIter': 10} -> 0.3680
 {'rank': 30, 'regParam': 0.2, 'maxIter': 5} -> 0.4304
 {'rank': 30, 'regParam': 0.2, 'maxIter': 10} -> 0.4215
Best params: {'rank': 30, 'regParam': 0.05, 'maxIter': 10}
Test RMSE (best model): 0.3327 | constant-mean baseline: 0.5000

=== ALS top-10 recommendations ===
42637   42636,42631,42626,42622,42634,42624,42632,42625,42638,42627
7381    7390,7404,7394,15647,7389,7385,2686,7408,19,24519
1677    29379,2678,29260,21843,2687,21556,2641,28263,2713,34278
49459   11800,46868,49488,49507,36380,49477,23764,33143,17155,49482
18227   18225,18224,7858,10812,18226,12222,966,16726,18917,15471
16246   4069,11072,16215,29226,16242,23185,16257,16194,16224,16259
14805   18005,23484,14823,17198,6611,10701,40610,4498,40850,7029
9249    9247,9279,9248,9261,9260,9251,9250,9275,9258,9259
49124   49108,49168,49153,49128,49115,49123,49116,49114,49148,14643
6794    6713,6716,6703,6723,6733,6730,6702,6700,6754,6770

## 5. Reproducibility

```
spark-submit --driver-memory 8g friend_recommend.py 42
spark-submit --driver-memory 8g als_recommend.py 42
```

- Run Part 1 first; it writes `mapreduce_recommendations.tsv`, which Part 2 reads for the comparison.
- The dataset is downloaded to the working directory on first use; no absolute paths are used.
- All randomness is seeded (sampling, split, negative sampling, CV, ALS). ALS results can still
  differ slightly across cluster configurations because of parallel floating-point summation.

## 6. Limitations

- Training on a 2-hop subgraph is a deliberate scale-down, not the full 4.8M-user graph.
- RMSE measures rating reconstruction, not ranking quality; overlap and mutual-friend averages are
  used as ranking-oriented complements.