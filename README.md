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

### Set covering (synthetic)

#### Data Pre-Process

1. run ```_1_generate_lp_data_mixed.py``` to generate graphs for set covering. Here the generation follows that in [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://github.com/ds4dm/learn2branch). We include our generated 1000 samples in /data/primal_format/m200n400_mixed2/.

2. run ```_2_infer_scip_primal.py``` to get a set of primal solution/slack to the original problems. We include the primal solutions/slacks for our generated samples in /data/primal_format/m200n400_mixed2/

3. run ```_3_generate_dual.py``` to get the dual format of linear relaxation to the original problems. We include the dual format of our generated samples in /data/dual_format/m200n400_mixed2/

4. run   ```_4_1_infer_scip_dual_solution.py``` and ```_4_2_infer_scip_dual_slack.py``` to infer a set of solutions / slack variables for the dual format of linear relaxation. We include them in /data/dual_solution(dual_slack)/m200n400_mixed2/
5. run ```_5_normalize_nodes.py``` to normalize the node inputs of the dataset
6. run ```_6_normalize_weight.py``` to normalize the edge weight in the dataset
7. run ```_7_generate_vcgraph.py``` to generate the VC-bipartite graph in torch-geometric in-memory dataset.

