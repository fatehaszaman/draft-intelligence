# Algorithm guide

Selected numerical routines are described as implemented. Fixed-width
arithmetic is assumed; algorithmic cost does not establish the quality of
historical labels, calibration, or out-of-sample predictions.

## Nearest historical comparisons

Implementation: [`HistoricalCompFinder.find_comps`](../scouting/historical_comps.py).

```text
# Historical Comparisons / Normalize + Full Distance Sort
# Input: query with D features; stored N-by-D training matrix; requested K matches
# Output: up to K finite-distance comparison records
# Time: O(N*D + N*log N + K*log K), typical comparison-sort model
# Memory: O(N*D + N + K) temporary/output storage, in addition to stored matrix

BUILD query features
NORMALIZE with stored training minima and ranges
diff = stored normalized features - query
distance = SQRT(row-wise SUM(diff * diff))
IF position filter is supplied:
    replace nonmatching distances with infinity
indices = ARGSORT(all distances), then take first K
FOR each selected finite distance:
    BUILD record with similarity = MAX(0, ROUND(100 - 50*distance))
RETURN records sorted by similarity descending
```

The position filter does not avoid computing distances for other positions.
The function fully sorts all N distances, rather than maintaining a K-sized
heap; NumPy's selected sorting backend determines strict worst-case behavior.
The broadcast difference is materialized, so workspace is not just O(K).

Watch: K=0 falls back to the configured default via `n or self.k`; negative
values use Python slice semantics rather than validation. No matching position
can produce an empty result. Similarity is a heuristic score, not a probability.

## One-vs-rest logistic training

Implementation: [`LogisticRegressionGD.fit`](../scouting/historical_comps.py).

```text
# Logistic Regression / Fixed-Iteration Gradient Descent
# Input: N rows, D features, C classes, I iterations
# Output: C-by-(D+1) weights and C loss histories
# Time: O(C*I*N*(D+1))
# Memory: O(N*(D+1) + C*(D+1) + C*I + N)

Xb = COPY features with a leading bias column
FOR each class:
    target = one-vs-rest binary labels
    theta = zeros
    REPEAT I times:
        p = sigmoid(CLIP(Xb @ theta))
        APPEND binary cross-entropy plus non-bias L2 penalty
        gradient = Xb.T @ (p-target) / N
        ADD non-bias L2 gradient
        theta = theta - learning_rate * gradient
    STORE theta
RETURN model
```

All I iterations run; loss is recorded, not used for early stopping.
Constructing `HistoricalCompFinder` performs this training before queries.
The prediction method divides sigmoid outputs by their row sum; this is not
softmax and normalization alone does not demonstrate calibration. Empty
training data and nonfinite features need separate validation.
