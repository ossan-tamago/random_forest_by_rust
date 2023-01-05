mod random_forest;

use random_forest::RandomForest;


fn main() {

    let mut rf = RandomForest::new();
    let X = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
    let y = vec![0, 1, 2];
    rf.fit(&X, &y, 10, 10, 2);
    let X_test = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
    let y_pred = rf.predict(&X_test);
    println!("{:?}", y_pred);

}