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
    with parallel_backend("dask"):
        tree_model = ParallelizedBinaryCategoricalDiscriminator("Y", max_depth = max_depth, min_sample = min_sample, nested_parallel = True)
        tree_model.train(df_train, XNAME_LST, NUMERIC_COLS)
    tree_model.dump_tree_to_file(fname)
    tree_model.save_tree(model_name)

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
    
#    tree_model.print_tree()
