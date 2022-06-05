# Parallelized Binary Cohort Detector
## Background
Apart from data modelings that solely aim to maximize the accuracy, data scientists more often need to come up with a non-trivial and <strong>interpretable</strong> data analysis to present to the business leaders. Tree-based classifiers are highly favored by data scientists for the modeling, but they offer very little more than the `Shapley` analysis when it comes to interpretation. To help business leaders identify cohorts of interest along with their profiles, we designed a pipeline that builds on the idea of a decision tree classifier, but it improves on a few aspects:
* Nested-parallelized algorithm that can leverage distributed computing power.
* Objective function that caters to the imbalanced nature of datasets.
* Output format easy to read and easy to process automatically.
* Built-in model evaluation pipeline that allows data scientist to pull out commonly used metrics with minimal efforts.

## Example Usage
An example of applying this pipeline to a binary classification problem is demonstrated as follows. First, we setup the process by initializing some parameters and fire up the `dask` clusters:
```Python
from parallelized_binary_discriminator import *

max_cpu = 3
min_sample = 2000
max_depth = 3

suffix = f"depth={max_depth}_minsample={min_sample}_demo"
fname = f"Results/info_tree_{suffix}.txt"
eval_fname = f"Results/eval_info_tree_{suffix}.txt"
model_name = f"Models/model_{suffix}.pkl"

df_train = pd.read_csv("Data/train_data.csv")
df_test = pd.read_csv("Data/test_data.csv")

XNAME_LST = ["C1", "C2", "X2"]
NUMERIC_COLS = ["X2"]

if __name__ == '__main__':
    dask.config.set({'distributed.comm.timeouts.connect': '120s'})
    try:
        cluster.shutdown()
        client.close()
    except:
        pass
    cluster = LocalCluster(n_workers=max_cpu, processes=True, threads_per_worker=2)
    client = Client(cluster, asynchronous=True)
```
Train and save the model:
```Python
with parallel_backend("dask"):
    tree_model = ParallelizedBinaryCategoricalDiscriminator("Y", max_depth = max_depth, min_sample = min_sample, nested_parallel = True)
    tree_model.train(df_train, XNAME_LST, NUMERIC_COLS)
tree_model.dump_tree_to_file(fname)
tree_model.save_tree(model_name)
```
Evaluate models on the training and test sets:
```Python
with open(model_name, "rb") as f:
    tree_model = pickle.load(f)
with open(fname, "a") as f:
    accuracy, recall, precision, f1_score = tree_model.evaluate(df_train)
    line = f"\nAccuracy = {round(accuracy * 100, 2)}%, Recall = {round(recall * 100, 2)}%, Precision = {round(precision * 100, 2)}%, F1-Score = {round(f1_score, 2)}"
    f.write(line)
tree_eval = tree_model.eva(df_test)
tree_model.dump_tree_to_file(eval_fname, tree = tree_eval)
with open(eval_fname, "a") as f:
    accuracy, recall, precision, f1_score = tree_model.evaluate(df_test)
    line = f"\nAccuracy = {round(accuracy * 100, 2)}%, Recall = {round(recall * 100, 2)}%, Precision = {round(precision * 100, 2)}%, F1-Score = {round(f1_score, 2)}"
    f.write(line)
```
Print out the results into the console:
```Python
tree_model.print_tree()
```
