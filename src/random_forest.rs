use rand::seq::SliceRandom;
use rand::thread_rng;

// 決定木のノードを表す構造体
struct DecisionTreeNode {
    feature: Option<usize>, // 特徴量のインデックス
    threshold: Option<f64>,  // 閾値
    left: Option<Box<DecisionTreeNode>>,  // 左側のノード
    right: Option<Box<DecisionTreeNode>>, // 右側のノード
    value: Option<f64>, // 葉ノードにおける予測値
}

// ランダムフォレストを表す構造体
pub struct RandomForest {
    trees: Vec<DecisionTreeNode>, // 決定木の集合
    n_classes: usize, // クラス数
}
impl RandomForest {
    // ランダムフォレストをトレーニングデータから生成する関数
    pub fn fit(
        &mut self,
        X: &Vec<Vec<f64>>,
        y: &Vec<usize>,
        n_trees: usize, // 生成する決定木の数
        max_depth: usize, // 決定木の深さの最大値
        min_samples_split: usize, // 分割を試みる最小サンプル数
    ) {
        let n_samples = X.len();
        let n_features = X[0].len();
        self.n_classes = *y.iter().max().unwrap() + 1;
        self.trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            // トレーニングデータからランダムなサブセットを取り出す
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            let mut X_sub: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
            let mut y_sub: Vec<usize> = Vec::with_capacity(n_samples);
            for i in indices.iter().take(n_samples) {
                X_sub.push(X[*i].clone());
                y_sub.push(y[*i]);
            }
            // サブセットから決定木を生成する
            let mut root = DecisionTreeNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: None,
            };
            Self::build_tree(&self,
                             &mut root,
                             &X_sub,
                             &y_sub,
                             0,
                             max_depth,
                             min_samples_split,
            );
            self.trees.push(root);
        }
    }
    // 決定木を生成する関数
    fn build_tree(&self,
                  node: &mut DecisionTreeNode,
                  X: &Vec<Vec<f64>>,
                  y: &Vec<usize>,
                  depth: usize,
                  max_depth: usize,
                  min_samples_split: usize,
    ) {
        let n_samples = X.len();
        let n_features = X[0].len();
        let mut class_counts = vec![0; self.n_classes];
        for y_i in y.iter() {
            class_counts[*y_i] += 1;
        }
        // 全てのサンプルが同じクラスに属する場合、葉ノードを生成する
        if class_counts.iter().all(|&x| x == 0) {
            let class_idx = y[0];
            node.value = Some(class_idx as f64);
            return;
        }
        // 深さの最大値を超える、あるいは分割を試みる最小サンプル数を下回る場合、葉ノードを生成する
        if depth >= max_depth || n_samples < min_samples_split {
            let class_idx = class_counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, count)| count)
                .unwrap()
                .0;
            node.value = Some(class_idx as f64);
            return;
        }
        let mut feature_idx = 0;
        let mut threshold = 0.0;
        let mut max_gain = 0.0;
        // 信息利得が最大になる特徴量と閾値を探す
        for i in 0..n_features {
            let values: Vec<f64> = X.iter().map(|x| x[i]).collect();
            let mut values_and_labels: Vec<(f64, usize)> = values
                .iter()
                .zip(y.iter())
                .sorted_by(|a, b| a.0.partial_cmp(b.0).unwrap())
                .collect();
            let mut current_entropy = Self::calculate_entropy(
                &class_counts,
                class_counts.iter().sum&(),
            );
            let mut n_left = 0;
            let mut n_right = class_counts.iter().sum();
            for j in 0..n_samples - 1 {
                let label = values_and_labels[j].1;
                class_counts[label] -= 1;
                n_left += 1;
                n_right -= 1;
                let left_entropy = Self::calculate_entropy(
                    &class_counts,
                    n_left,
                );
                let right_entropy = Self::calculate_entropy(
                    &class_counts,
                    n_right,
                );
                let gain = current_entropy
                    - (n_left as f64 / n_samples as f64) * left_entropy
                    - (n_right as f64 / n_samples as f64) * right_entropy;
                if gain > max_gain {
                    feature_idx = i;
                    threshold = (values_and_labels[j].0 + values_and_labels[j + 1].0) / 2.0;
                    max_gain = gain;
                }
            }
        }
        // 分割する
        node.feature = Some(feature_idx);
        node.threshold = Some(threshold);
        let mut left_X = Vec::new();
        let mut left_y = Vec::new();
        let mut right_X = Vec::new();
        let mut right_y = Vec::new();
        for i in 0..n_samples {
            if X[i][feature_idx] < threshold {
                left_X.push(X[i].clone());
                left_y.push(y[i]);
            } else {
                right_X.push(X[i].clone());
                right_y.push(y[i]);
            }
        }
        // 左側のノードを生成する
        if left_X.len() > 0 {
            let mut left = DecisionTreeNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: None,
            };
            Self::build_tree(&self,
                             &mut left,
                             &left_X,
                             &left_y,
                             depth + 1,
                             max_depth,
                             min_samples_split,
            );
            node.left = Some(Box::new(left));
        }
        // 右側のノードを生成する
        if right_X.len() > 0 {
            let mut right = DecisionTreeNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: None,
            };
            Self::build_tree(&self,
                             &mut right,
                             &right_X,
                             &right_y,
                             depth + 1,
                             max_depth,
                             min_samples_split,
            );
            node.right = Some(Box::new(right));
        }
    }
    // エントロピーを計算する関数
    fn calculate_entropy(class_counts: &Vec<usize>, n: usize) -> f64 {
        let mut entropy = 0.0;
        for count in class_counts.iter() {
            if *count > 0 {
                let p = *count as f64 / n as f64;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
    // ランダムフォレストを使って予測する関数
    pub fn predict(&self, X: &Vec<Vec<f64>>) -> Vec<usize> {
        let n_samples = X.len();
        let mut y_pred = vec![0; n_samples];
        for i in 0..n_samples {
            let mut class_counts = vec![0; self.n_classes];
            for tree in self.trees.iter() {
                Self::count_classes(tree, &X[i], &mut class_counts);
            }
            let mut max_count = 0;
            let mut max_class = 0;
            for (j, count) in class_counts.iter().enumerate() {
                if *count > max_count {
                    max_count = *count;
                    max_class = j;
                }
            }
            y_pred[i] = max_class;
        }
        y_pred
    }

    // 決定木を走査し、各クラスの数を数える関数
    fn count_classes(
        node: &DecisionTreeNode,
        x: &Vec<f64>,
        class_counts: &mut Vec<usize>,
    ) {
        match node.value {
            Some(value) => {
                class_counts[value as usize] += 1;
            }
            None => {
                let feature = node.feature.unwrap();
                let threshold = node.threshold.unwrap();
                if x[feature] < threshold {
                    Self::count_classes(
                        node.left.as_ref().unwrap(),
                        x,
                        class_counts,
                    );
                } else {
                    Self::count_classes(
                        node.right.as_ref().unwrap(),
                        x,
                        class_counts,
                    );
                }
            }
        }
    }
}