# SynTME: A Tumor Microenvironment-Aware, Pharmacology-Inspired Multi-Stage Framework for Drug Synergy Prediction

## 📌 Introduction
Combination therapy mitigates toxicity and resistance, yet experimental screening
remains costly and inefficient. Although machine learning has advanced drug synergy prediction, most existing models are trained on in vitro monolayer cell-line data that omit tumor microenvironment information, limiting clinical
relevance. Furthermore, these methods typically rely on static feature fusion and do not organize drug response in a stage-wise manner that reflects pharmacodynamically motivated aspects of synergy formation within a single treatment context. We aim to address these limitations by proposing
SynTME, a TME-aware and pharmacology-inspired framework in which the TME is operationalized through immune
infiltration–based descriptors.

## 🗂️ Repository Architecture
The SynTME repository is structured to facilitate reproducibility and future extensions:
* `data/`: Contains raw and processed data manifolds (requires manual download).
* `dataset/`: Core scripts for topological dataset construction (`syntme_dataset.py`).
* `models/`: Architectural components of the SynTME framework (`syntme_core.py`, `head.py`, `model_utlis.py`).
* `experiment/`: Directory for automated logging, model checkpoints, and inference outputs.
* `utils.py`: Subroutines for data stream loading, metrics calculation, and process flow control.
* `metrics.py`: Statistical evaluation metrics computation.
* `main.py`: The primary execution protocol for the SynTME framework.

## 📦 Dataset and Software Information

The primary benchmark dataset used in this study is DrugComb v1.5. Additional benchmark datasets include the O'Neil oncology combination screen, Oncology Screen/NCI ALMANAC, and DrugCombDB. Cell-line molecular features were derived from DepMap genomic profiles. Drug chemical information and SMILES annotations were obtained from ChEMBL. Immune infiltration descriptors were constructed from TCGA immune infiltration resources together with UCSC Xena phenotype annotations.

The SynTME framework relies on PyTorch and PyTorch Geometric for dynamic graph and tensor operations. We recommend isolating the environment using Conda.

Execute the following commands to establish the required dependencies:

```bash
conda create -n syntme_env python=3.8 pip
conda activate syntme_env

# Install PyTorch and core dependencies (adjust CUDA version as dictated by your hardware)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install torch-geometric
pip install pandas tqdm scikit-learn rdkit
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

## 🔁 Reproducibility Settings

The reported sample-level train/validation/test partitions were generated using split seed `42`. The reported cancer-type-constrained TME allocation was generated using sampling seed `42`.

Model-training variability was evaluated over five independent runs with training seeds `2025`, `36`, `2002`, `42`, and `3660`. The corresponding performance metrics are reported as mean ± standard deviation where applicable.

The same data splits and TME sampling protocol were used across compared models unless otherwise specified.


## 🧬 Implementation Details

### Data Manifold Preparation
1. **Feature Engineering**: SynTME operates on normalized multi-omics data configurations and extracted molecular substructure encodings. The required tensor arrays should be located in `data/0_cell_data/` and `data/1_drug_data/`. 
   
   > **Important Notice regarding Data Access**: 
   > Due to GitHub's storage constraints for large binary files, the pre-processed `.npy` arrays are hosted externally. Please acquire the necessary datasets from [https://pan.baidu.com/s/1vATeNX0XhzjLkN47mydHFw 提取码: tej7] and populate the `data/` directory accordingly.

2. **Combination Registry**: The primary execution targets are stored within `data/split/all_items.npy`. The required dimensionality for each entry is `[Mol_A_SMILES, Mol_B_SMILES, Context_ID, Synergy_Score]`. For inference tasks on unlabelled pairs, initialize the target score to `0.0`.
3. Ensure that all static paths within `utils.py` correctly point to your local data storage prior to execution.
After downloading the external data package, the `data/` directory should follow the structure below（The `split/` directory contains the sample-level train/validation/test partitions used for the reported experiments, while `independent_dataset/` contains the processed inputs for the additional benchmark datasets.）:
```text
data/
├── 0_cell_data/
│   ├── 4079g/
│   └── cellTme.npy
├── 1_drug_data/
│   └── drugSmile_drugSubEmbed_2644.npy
├── independent_dataset/
│   ├── indep0-oneil/
│   ├── indep1-OncologyScreen/
│   └── indep2-DrugCombDB/
├── raw_data/
│   ├── cell_info.csv
│   ├── drug_info.csv
│   ├── gene_4079.csv
│   └── drug_substructure/
├── processed/
└── split/
```
### Phase 1: Framework Optimization (Training)
To initialize the training protocol and optimize the SynTME framework, execute the standard training command. Logs will be automatically generated and timestamped in the `./experiment` directory.

```bash
python main.py \
    --mode train \
    --celldataset 2 \
    --cellencoder cellCNNTrans \
    --batch_size 32 \
    --epochs 500 \
    --cv_splits 0 > './experiment/syntme_train_'$(date +'%Y%m%d_%H%M').log
```

### Phase 2: Performance Validation (Testing)
To evaluate an optimized checkpoint against the designated test manifold, invoke the test mode. You must explicitly declare the path to your trained weights via the --saved_model argument.

```
python main.py \
    --mode test \
    --saved_model ./experiment/syntme_0_best_test.pth \
    --device cuda:0 > './experiment/syntme_test_'$(date +'%Y%m%d_%H%M').log

```
### Phase 3: Synergy Inference & Topology Extraction
For novel combinatorial predictions where empirical synergy labels are absent, utilize the infer mode. This protocol not only computes the predicted synergy logits but can also export the latent attention topologies for downstream interpretability analysis.

Set --output_attn 1 to extract the cross-modal attention matrices.
```

python main.py \
    --mode infer \
    --infer_path ./data/split/sample_infer_items.npy \
    --saved_model ./experiment/syntme_0_best_test.pth \
    --output_attn 1 > './experiment/syntme_infer_'$(date +'%Y%m%d_%H%M').log
```
