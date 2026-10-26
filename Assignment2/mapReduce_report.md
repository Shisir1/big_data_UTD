# Friend Recommendation via MapReduce (Spark RDD)

## Idea

For a candidate pair (s, c) that are not friends, the number of mutual friends is the
number of users m whose friend list contains **both** s and c.
So each user m, looking at its own friend list L, "votes" once for every pair (s, c) in L.
Summing the votes per pair gives the mutual-friend count.

## Key optimization

The textbook version emits every pair in every friend list: about sum(d^2) records
(d = list length), which blows up on high-degree users. Only 10 users need recommendations,
so we emit only pairs whose first element is a sampled user. Cost drops to about sum(d)
records, since each list contains few sampled users.

## Pseudocode

```
INPUT: lines "uid \t f1,f2,..."

# Stage 0 - symmetric adjacency
adj      = lines.parse().flatMap(edge -> (u,f), (f,u)).distinct().groupByKey()  # (user, sorted friends)
sample   = random 10 keys of adj
direct   = adjacency lists of the sampled users (tiny, broadcast)

# Stage 1 - MAP
map (m, L):
    for s in L where s in sample:
        for c in L:
            if c != s and c not in direct[s]:
                emit ((s, c), 1)

# Stage 2 - REDUCE
reduceByKey((s, c), [1,1,...]) -> ((s, c), sum)

# Stage 3 - regroup and rank
map ((s, c), n) -> (s, (c, n))
groupByKey s -> sort by (n desc, c asc) -> take first 10 -> [c1, c2, ...]

OUTPUT: "s \t c1,c2,...,c10"   (fewer than 10 if fewer candidates exist)
```

## Details

- **Undirected edges:** each edge is added in both directions and deduplicated, so the result
  does not depend on the input being symmetric.
- **Exclusion of existing friends:** done in the map via the broadcast `direct` sets, which
  shrinks the shuffle.
- **Tie-break:** equal counts are ordered by smaller UserID first (deterministic).
- **Users with no candidates:** printed with an empty list.
- **Complexity:** map output is about sum over m of (sampled users in L(m)) * |L(m)|; one shuffle for
  `reduceByKey`, one tiny shuffle for the final grouping.