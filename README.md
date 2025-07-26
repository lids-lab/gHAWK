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

All of our node‐classification experiments build on the official OGB baseline code, with custom feature injections (Bloom filters, TransE embeddings, word2vec/RoBERTa embeddings). We compare directly against OGB’s reported results, using the same training/validation splits and evaluation protocol.

### Benchmarks & Models

- **OGBN-MAG**  
  - Models: RGCN, GraphSAGE, GraphSAINT, ClusterGCN  

- **OGBN-MAG240**  
  - Models: RGCN (RGraphSAGE & RGAT variants), GraphSAGE, GAT

### Feature Options (`--feature_type`)

| Option                     | Dataset      | Description                                                                    |
|----------------------------|--------------|--------------------------------------------------------------------------------|
| `noFeature`                | MAG & MAG240 | No extra features (all-zeros)                                                  |
| `word2vec`                 | MAG          | Paper Word2Vec embeddings (`data.x_dict['paper']`, 128-dim)                   |
| `bloom`                    | MAG & MAG240 | Bloom-filter vectors only                                                      |
| `roberta`                  | MAG240       | Pretrained RoBERTa embeddings (first 768 dims)                                 |
| `roberta+bloom+transe`     | MAG240       | Concatenation of RoBERTa, Bloom & TransE representations  
> **Bloom projection:**  
> Automatically detects `bloom_dim = bloom.shape[1]` and applies  

### Running a Training Job

1. **Prepare data**  
   Place your dataset under `ROOT/ogbn-mag` or `ROOT/mag240m` and run:
   ```bash
   python -c "from ogb.lsc import MAG240MDataset; MAG240MDataset('ROOT/ogbn-mag')"
   ```
2. **Launch training**  
   From ROOT/ogbn-mag240-gnns/, just invoke the desired script. For example:
   ```bash
   python RGCN.py \
  --data_dir ROOT/mag240m \
  --feature_type roberta+bloom+transe \
  --model rgraphsage \
  --sizes 25-15 \
  --device 0,1
   ```
Or you can hard‐code defaults at the top of each *.py (e.g., FEATURE_TYPE = "bloom").
3. **Multi‐GPU support**  
   By default the scripts use a single CUDA device; to use multiple GPUs, pass comma‐separated device IDs:
   ```bash
   --device 0,1,2
   ```

## Link Prediction
