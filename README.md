# Random Forest

Rustによるランダムフォレストの実装です。

## 使い方

train関数で学習を行います。

```rust
let random_forest = RandomForest::train(&features, &labels, 10, 2);
```

predict関数で予測を行います。

```rust
let prediction = random_forest.predict(&features);
```

