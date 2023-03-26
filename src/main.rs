use rand::prelude::*;
use std::collections::HashMap;

type Label = u8;
type Feature = f32;

#[derive(Clone)]
struct DecisionTree {
    feature_idx: usize,
    threshold: Feature,
    left: Option<Box<DecisionTree>>,
    right: Option<Box<DecisionTree>>,
    class_counts: HashMap<Label, usize>,
}

impl DecisionTree {
    fn train(
        features: &Vec<Vec<Feature>>,
        labels: &Vec<Label>,
        depth: usize,
        max_depth: usize,
        rng: &mut ThreadRng,
    ) -> Self {
        let sample_size = (features.len() as f32).sqrt() as usize;
        let mut subset = Vec::new();
        let mut subset_labels = Vec::new();
        for _ in 0..sample_size {
            let idx = rng.gen_range(0..features.len());
            subset.push(features[idx].clone());
            subset_labels.push(labels[idx]);
        }

        let mut class_counts = HashMap::new();
        for &label in subset_labels.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        if depth >= max_depth || class_counts.len() == 1 {
            return DecisionTree {
                feature_idx: 0,
                threshold: 0.0,
                left: None,
                right: None,
                class_counts,
            };
        }

        let num_features = features[0].len();
        let mut best_feature_idx = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;
        for feature_idx in 0..num_features {
            let mut feature_values = Vec::new();
            for i in 0..sample_size {
                feature_values.push(subset[i][feature_idx]);
            }
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut class_counts_left = HashMap::new();
            let mut class_counts_right = class_counts.clone();
            let mut num_left = 0;
            let mut num_right = sample_size;
            for i in 0..(sample_size - 1) {
                let label = subset_labels[i];
                *class_counts_left.entry(label).or_insert(0) += 1;
                *class_counts_right.entry(label).or_insert(0) -= 1;
                num_left += 1;
                num_right -= 1;
                if feature_values[i] == feature_values[i + 1] {
                    continue;
                }
                let left_entropy = entropy(&class_counts_left);
                let right_entropy = entropy(&class_counts_right);
                let entropy_sum = left_entropy * (num_left as f32) + right_entropy * (num_right as f32);
                let gain = entropy_sum - entropy(&class_counts);
                if gain > best_gain {
                    best_feature_idx = feature_idx;
                    best_threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;
                    best_gain = gain;
                }
            }
        }

        if best_gain == 0.0 {
            return DecisionTree {
                feature_idx:
                0,
                threshold: 0.0,
                left: None,
                right: None,
                class_counts,
            };
        }

        let mut subset_left = Vec::new();
        let mut subset_left_labels = Vec::new();
        let mut subset_right = Vec::new();
        let mut subset_right_labels = Vec::new();
        for i in 0..sample_size {
            if subset[i][best_feature_idx] <= best_threshold {
                subset_left.push(subset[i].clone());
                subset_left_labels.push(subset_labels[i]);
            } else {
                subset_right.push(subset[i].clone());
                subset_right_labels.push(subset_labels[i]);
            }
        }

        let left = if subset_left.is_empty() {
            None
        } else {
            Some(Box::new(DecisionTree::train(
                &subset_left,
                &subset_left_labels,
                depth + 1,
                max_depth,
                rng,
            )))
        };
        let right = if subset_right.is_empty() {
            None
        } else {
            Some(Box::new(DecisionTree::train(
                &subset_right,
                &subset_right_labels,
                depth + 1,
                max_depth,
                rng,
            )))
        };

        DecisionTree {
            feature_idx: best_feature_idx,
            threshold: best_threshold,
            left,
            right,
            class_counts,
        }
    }

    fn predict(&self, features: &Vec<Feature>) -> Label {
        if let Some(left) = &self.left {
            if features[self.feature_idx] <= self.threshold {
                return left.predict(features);
            }
        }
        if let Some(right) = &self.right {
            return right.predict(features);
        }
        *self
            .class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .unwrap()
            .0
    }
}

fn entropy(class_counts: &HashMap<Label, usize>) -> f32 {
    let mut total_count = 0;
    let mut ent = 0.0;
    for &count in class_counts.values() {
        total_count += count;
    }
    for &count in class_counts.values() {
        let p = count as f32 / total_count as f32;
        ent -= p * p.log2();
    }
    ent
}

struct RandomForest {
    trees: Vec<DecisionTree>,
}

impl RandomForest {
    fn train(
        features: &Vec<Vec<Feature>>,
        labels: &Vec<Label>,
        num_trees: usize,
        tree_depth: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut trees = Vec::new();
        for _ in 0..num_trees {
            trees.push(DecisionTree::train(
                features,
                labels,
                0,
                tree_depth,
                &mut rng,
            ));
        }
        RandomForest { trees }
    }
    fn predict(&self, features: &Vec<Feature>) -> Label {
        let mut total_prediction = 0;
        for tree in self.trees.iter() {
            total_prediction += tree.predict(features) as i32;
        }
        if total_prediction as f32 / self.trees.len() as f32 > 0.5 {
            1
        } else {
            0
        }
    }
}

fn main() {
// データの準備
    let features = vec![
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![0.5, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 0.5],
        vec![1.0, 0.5],
        vec![0.0, 1.0],
    ];
    let labels = vec![1, 0, 1, 0, 0, 1, 0];

// ランダムフォレストの訓練
    let random_forest = RandomForest::train(&features, &labels, 10, 2);

// 予測
    let prediction = random_forest.predict(&vec![0.7, 0.3]);
    println!("Prediction: {}", prediction);
}