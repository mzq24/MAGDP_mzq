from torch_geometric.nn import GATv2Conv, GATConv, TransformerConv, GCNConv

def init_GNNConv(args):
    if args['gnn_conv'] == 'GATv2Conv':
        gnnconv = GATv2Conv
    elif args['gnn_conv'] == 'GATConv':
        gnnconv = GATConv
    elif args['gnn_conv'] == 'TransformerConv':
        gnnconv = TransformerConv
    elif args['gnn_conv'] == 'GCNConv':
        gnnconv = GCNConv
    return gnnconv

def build_network(args, gnnconv):
    '''
    pipeline:   
        1: raw_emb (Dyn) -> C2A (CCLs) -> A2A (Int)
        2: raw_emb (Dyn) -> A2A (Int)
        3: raw_emb (Dyn)
        4: raw_emb (Dyn) -> C2A (CCLs)

    norm_seg:
        1: Dyn, CCLs, Int,  seg_number = 3,  
        2: Dyn, Int         seg_number = 2, 
        3: Dyn              seg_number = 1, 
        4: Dyn, CCLs        seg_number = 2, 

    C2A: CCL_to_Agn_GAT_1 
        input: from raw_emb, 
        dim: 
            1, 4: enc_hidden_size -> enc_gat_size * num_gat_head

    A2A, Agn_to_Agn_GAT_1
        input: 
            1: from C2A
            2: from raw_emb
        dim: 
            1: enc_gat_size * num_gat_head -> enc_gat_size * num_gat_head
            2: enc_hidden_size -> enc_gat_size * num_gat_head
    '''

    # GCN is different from other GNNs, such as GAT, GATv2, and Transformer
    # GCN output dim with no heads, while other GNNs output dim need to product heads
    if args['gnn_conv'] == 'GCNConv':
        CCL_to_Agn_GAT_1  = gnnconv(args['enc_hidden_size'], args['enc_gat_size']*args['num_gat_head'])
        if args['norm_seg'] == 2:
            Agn_to_Agn_GAT_1  = gnnconv(args['enc_hidden_size'], args['enc_gat_size']*args['num_gat_head'])
        else:
            Agn_to_Agn_GAT_1 = gnnconv(args['enc_gat_size']*args['num_gat_head'], args['enc_gat_size']*args['num_gat_head'])
    
    # others         
    else:
        # C2A
        CCL_to_Agn_GAT_1  = gnnconv(args['enc_hidden_size'], args['enc_gat_size'], 
                            heads=args['num_gat_head'], concat=True, 
                            add_self_loops=False,
                            dropout=0.0)
        # A2A
        if args['norm_seg'] == 2:
            Agn_to_Agn_GAT_1  = gnnconv(args['enc_hidden_size'], args['enc_gat_size'], 
                            heads=args['num_gat_head'], concat=True,  
                            add_self_loops=False,
                            dropout=0.0)
        # elif args['norm_seg'] == 1:
        else:
            Agn_to_Agn_GAT_1  = gnnconv(args['enc_gat_size']*args['num_gat_head'], args['enc_gat_size'], 
                            heads=args['num_gat_head'], concat=True,  
                            add_self_loops=False,
                            dropout=0.0)
    
    return CCL_to_Agn_GAT_1, Agn_to_Agn_GAT_1