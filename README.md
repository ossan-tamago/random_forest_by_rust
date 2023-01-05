# Random Forest

Rustによるランダムフォレストの実装です。

## 使い方

```rust
use random_forest::RandomForest;
```

fit関数で学習を行います。

```rust
let mut rf = RandomForest;
rf.fit(&x, &y, n_trees, max_depth, min_samples_split);
```

predict関数で予測を行います。

```rust
let y_pred = rf.predict(&x);
```

