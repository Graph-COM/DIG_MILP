# DIG_MILP
The official implementation of DIG-MILP: A Deep Instance Generator for Mixed-Integer Linear Programming with Feasibility Guarantee

## Environment Requirement

To implement the code, the environment below is required:
```
SCIP version 7.0.3
ecole       0.7.3
matplotlib          3.7.2
networkx            3.0
numpy               1.24.4
PySCIPOpt           3.5.0
PyYAML              6.0.1
scikit-learn        1.3.0
scipy               1.10.1
torch               2.0.0+cu117
torch-geometric     2.3.1
torchaudio          2.0.1+cu117
torchvision         0.15.1+cu117
tqdm                4.65.0
wandb               0.15.5
```
The corresponding CUDA version is 11.7 and CUDNN version is 8500.
To install [PYSCIPOpt](https://github.com/scipopt/PySCIPOpt), please first install [SCIP](https://scipopt.org/doc).

## Code implementation

We include the code implementation for all the four tasks (set covering, combinatorial auctions, CVS and IIS).

### Combinatorial auctions (CA) (synthetic)

#### Data pre-process

1. run ```_1_generate_lp_data_mixed.py``` to generate graphs for combinatorial auctions. Here the generation follows that in [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://github.com/ds4dm/learn2branch). 

2. run ```_2_infer_scip_primal.py``` to get a set of primal solution/slack to the original problems.

3. run ```_3_generate_dual.py``` to get the dual format of linear relaxation to the original problems.

4. run   ```_4_1_infer_scip_dual_solution.py``` and ```_4_2_infer_scip_dual_slack.py``` to infer a set of solutions / slack variables for the dual format of linear relaxation.
5. run ```_5_normalize_nodes.py``` to normalize the node inputs of the dataset.
6. run ```_6_normalize_weight.py``` to normalize the edge weight in the dataset.
7. run ```_7_generate_vcgraph.py``` to generate the VC-bipartite graph in torch-geometric in-memory dataset.
8. run```_8_normalize_degree.py``` to normalize the node degrees.

#### DIG-MILP training and inference
9. run ```sh 9_train.sh``` to train the model. The pre-trained DIG-MILP model is in the train_files.
10. run ```sh 10_generate.sh``` to generate new instances.

#### downstream task - parameter tuning without sharing data
enter /downstream_parameter/,

11. run ```sh 17_infer_log.sh``` and ```sh 17_infer_log2.sh``` to infer the 45 different hyper-parameter seeds for SCIP.
    run the script for three times and change the target output path to 'time_log0', 'time_log1', and 'time_log2'.
    
12. run ``paint_similarity.py``` to paint the Pearson correlation figure and get the score.

* We include the SCIP solution time for all the four problems in the time_log folder and the corresponding figure in the paint folder.

#### downstream task - train machine learning models
enter / downstream_train_ml/,
13. run ```_1_infer_generate_primal.py``` to get the optimal solution for the generated MILP instances.
14. run ```_2_generate_train_data.py``` to generate the training data for downstream task.
15. run ```_7_generate_val_lp.py``` and ```_8_generate_val_data.py``` to generate the validation data, run ```_4_generate_test_lp.py``` amd ```_5_generate_test_graph.py``` to generate the testing data.
16. run ```sh 3_train.sh``` to train the downstream machine learning model.
17. run ```_6_test.py``` to test the model.

* The implementation of Set covering (SC), CVS and IIS is similar to CA. But there could be some difference in data pre-processing:

### CVS and IIS
#### Data pre-process
To pre-process the data downloaded from [MIPLIB2017](https://miplib.zib.de/tag_benchmark.html), we provide the script to pre-process and we also directly include the pre-processed version (in primal_format folder). 




