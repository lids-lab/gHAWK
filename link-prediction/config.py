import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Hybrid model with Bloom+TransE+RotatE+PyG SAGEConv aggregator")
    parser.add_argument('--train_pt', type=str, default='/home/hxs7303/codes/wikikg-v2/wikikg-v2/split/time/train.pt',
                        help="Path to train data (head, relation, tail)")
    parser.add_argument('--valid_pt', type=str, default='/home/hxs7303/codes/wikikg-v2/wikikg-v2/split/time/valid.pt',
                        help="Path to valid data (head_neg, tail_neg)")
    parser.add_argument('--test_pt',  type=str, default='/home/hxs7303/codes/wikikg-v2/wikikg-v2/split/time/test.pt',
                        help="Path to test data (head_neg, tail_neg)")
    
    parser.add_argument('--bloom_out', type=str, default='/home/hxs7303/node_bloom_weighted_reltail.npy',
                        help="Path to bloom arrays (.npy file) [num_nodes, bloom_dim]")
    parser.add_argument('--transE_path', type=str,
                        default='/home/hxs7303/codes/Emb-Fatima/TransE/250-30.0/1735881913.4307704/entity_embedding.npy',
                        help="Path to TransE entity embeddings [num_nodes, dim]")
    parser.add_argument('--transE_rel_path', type=str,
                        default='/home/hxs7303/codes/Emb-Fatima/TransE/250-30.0/1735881913.4307704/relation_embedding.npy',
                        help="Path to TransE relation embeddings [num_relations, dim]")
    parser.add_argument('--bloom_size', type=int, default=500)
    parser.add_argument('--num_hashes', type=int, default=3)
    parser.add_argument('--proj_dim_bloom', type=int, default=256)
    parser.add_argument('--proj_dim_transe', type=int, default=256)
    parser.add_argument('--gnn_hidden_dim', type=int, default=256)
    parser.add_argument('--gnn_out_dim', type=int, default=256, help="Must be even for RotatE")
    parser.add_argument('--batch_size', type=int, default=6500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--num_relations', type=int, default=1070,
                        help="Number of relations (if using reversed edges, e.g. 1070; otherwise, 535)")
    parser.add_argument('--generate_bloom', action='store_true',
                        help="Generate bloom filters from scratch (if not, load from file)")
    

    parser.add_argument('--conv_type', type=str, default='sage', help="Choose between relation aware(rgcn) or Relation Agnositc (sage) setting default is ")
    parser.add_argument('--num_bases', type=int, default=30)
    
    
    
    return parser.parse_args()
