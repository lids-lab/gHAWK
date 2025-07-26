# gHAWK
gHAWK: Local and Global Neighborhood Aggregation for Representation Learning on Knowledge Graphs

## 🚀 Environment

- **Python** ≥ 3.8  
- **PyTorch** ≥ 1.12  
- **PyG** ≥ 2.x  
- **OGB** ≥ 1.3  
- **PyTorch Lightning** ≥ 1.9  
- **torchmetrics**, **numpy**, **tqdm**

Example installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torch-geometric ogb pytorch_lightning torchmetrics numpy tqdm
``` 

## Node Classification
### Heterogeneous GNNs on OGB Benchmarks: 
This repository implements several baseline and extended Graph Neural Networks for node classification on the OGBN-MAG and OGBN-MAG240 benchmarks. All models build upon the official OGB baseline code, injecting new feature sets (Bloom filters, TransE, word2vec/RoBERTa embeddings) to measure their impact.

---



## Link Prediction
