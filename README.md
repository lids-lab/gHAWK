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

All of our node‐classification experiments build on the official OGB baseline code, with custom feature injections (Bloom filters, TransE embeddings, RoBERTa representations). We compare directly against OGB’s reported results, using the same training/validation splits and evaluation protocol.

### Datasets & Model Suites

- **OGBN-MAG**  
  - Models:  
    - RGCN  
    - GraphSAGE  
    - GraphSAINT  
    - ClusterGCN  

- **OGBN-MAG240**  
  - Models:  
    - RGCN (RGraphSAGE and RGAT variants)  
    - GraphSAGE  
    - GAT  

---



## Link Prediction
