import torch_geometric.transforms as T


def auto_generate_metapaths(data, target_type=None, max_length=2):
    """
    Automatically search for all meta-paths in HeteroData that start 
    and end with target_type.
    """
    # Default to the first node type if not specified
    if target_type is None:
        target_type = data.node_types[0]
        
    valid_metapaths = []
    
    # Construct the schema adjacency list: {src_type: [(rel, dst_type), ...]}
    schema_adj = {}
    for src, rel, dst in data.edge_types:
        if src not in schema_adj: 
            schema_adj[src] = []
        schema_adj[src].append((rel, dst))
    
    def dfs(current_type, current_path, length):
        # Termination condition: reached maximum length
        if length == max_length:
            # If we have returned to the target_type, it represents a valid closed-loop meta-path
            if current_type == target_type:
                valid_metapaths.append(list(current_path))
            return

        if current_type in schema_adj:
            for rel, next_type in schema_adj[current_type]:
                # Get existing relation names in the current path
                existing_rels = {edge[1] for edge in current_path}
                
                # Avoid repeating the same relation type in a single path
                # (e.g., prevents 'cites' -> 'cites')
                if rel in existing_rels:
                    continue
                
                new_edge = (current_type, rel, next_type)
                dfs(next_type, current_path + [new_edge], length + 1)

    # Start DFS search
    dfs(target_type, [], 0)
    
    return valid_metapaths