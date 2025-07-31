import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from .utils import VNG_utils
MAX_SAMPLING_SIZE = 500
@torch.no_grad()
def softlabel_based_hard_nodes_sampling(x:torch.tensor,y,n_cls,diffusion_model,teacher,temperature, padding,guidance=7.5, hard_factor = 0.5, aug_mode = "ratio", over_sample_rate = 1, is_hard_sample = True, is_beta_sampling = True,device="cuda:0"):
    dis = VNG_utils.class_dis(y,n_cls)
    x = F.pad(x,pad=padding,mode="constant",value=0).to(device)
    n_emb = x.shape[1]
    soft_labels = teacher.softmax_with_temperature(x,temperature)
    hard_labels = y
    max_size = torch.max(dis)
    aug_size = copy.deepcopy(dis)
    if aug_mode == "ratio":
        assert over_sample_rate is not None, "If aug_mode is \"ratio\" then over_sample_rate must be given."
        aug_size[aug_size == max_size] = 0
        f_size = torch.tensor(aug_size*(over_sample_rate + 1),dtype=torch.int32)
        aug_size = torch.clip(f_size,max=max_size)-aug_size
    elif aug_mode == "mean":
        avrg = torch.sum(dis) // (n_cls)
        aug_size = avrg - aug_size
        aug_size[aug_size < 0] = 0
    elif aug_mode == "max":
        aug_size = max_size = aug_size
    else:
        raise ValueError("undefined \"aug_mode\"={}".format(aug_mode))

    # aug_size = aug_size * confidence_filter_ratio
    # assert confidence_filter_ratio > 1, "parameter confidence_filter_ratio must larger than 1"
    x0 = torch.tensor([],device=device)
    labels = torch.tensor([],device=device)
    sampling_src_idx = torch.tensor([],device=device)
    for class_,class_aug_size in enumerate(aug_size):
        if class_aug_size == 0 : continue
        print("Generating {} samples for class{}".format(int(class_aug_size),class_))
        node_classes = torch.full([int(class_aug_size)],fill_value=class_,device=device)
        # filter hard sample
        class_mask = (hard_labels == class_)
        class_indices = torch.nonzero(class_mask, as_tuple=True)[0]
        nodes_confidence = 1-torch.index_select(soft_labels[class_mask], dim = 1, index=torch.tensor(class_,device=device)).view(-1)
        if is_hard_sample:
            _,indices = torch.topk(nodes_confidence,(sum(class_mask) + 1)//2)
        else:
            indices = torch.ones((sum(class_mask),),dtype=torch.bool)
        src_indices = class_indices[indices]
        hard_samples = soft_labels[class_mask][indices]
        if is_beta_sampling:
            mean_confidence = torch.mean(hard_samples,dim=0)
            variance_confidence = mean_confidence / 100
            # print("mean ", mean_confidence)
            # print("var ", variance_confidence)
            # assume that confidence follow Beta districution
            alpha = mean_confidence * (mean_confidence * (1 - mean_confidence) / variance_confidence - 1)
            beta_param = (1 - mean_confidence) * (mean_confidence * (1 - mean_confidence) / variance_confidence - 1)
            # print("alpha ", alpha)
            # print("beta ", beta_param)
            confidence_sample = torch.distributions.Beta(alpha, beta_param).sample((class_aug_size,)).to(device)
            confidence_guidance = confidence_sample / torch.sum(confidence_sample,dim=1,keepdim=True)
        else:
            random_indices = torch.randint(0, hard_samples.shape[0]-1, (class_aug_size,))
            confidence_guidance = hard_samples[random_indices]
        # print(confidence_guidance)

        overall_guidance = confidence_guidance * (1-hard_factor) + F.one_hot(node_classes,n_cls) * hard_factor
        x_t = torch.randn([class_aug_size,1,n_emb]).to(device)
        if MAX_SAMPLING_SIZE < 500:
            x0_v,_ = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
                                                ,x_t=x_t,y=overall_guidance,padding=padding,save_frames=False,device=device)
        else:
            x0_v = []
            for step in range(0,class_aug_size // MAX_SAMPLING_SIZE+1):
                begin = step * MAX_SAMPLING_SIZE
                end = (step + 1) * MAX_SAMPLING_SIZE if (step + 1) * MAX_SAMPLING_SIZE < class_aug_size else class_aug_size
                x0_b,_ = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
                                                ,x_t=x_t[begin:end],y=overall_guidance[begin:end],padding=padding,save_frames=False,device=device)
                x0_v.append(x0_b)
            x0_v = torch.cat(x0_v, dim=0)
            
        # print("generated node feature quility:")
        # logits = teacher(torch.detach(x0_v).squeeze(1))
        # preds = logits.argmax(1)
        # print(preds)
        if x0.numel() == 0:
            x0 = copy.deepcopy(x0_v)
            labels = copy.deepcopy(node_classes)
            sampling_src_idx = copy.deepcopy(src_indices)
        else:
            x0 = torch.concat([x0,x0_v],dim=0)
            labels = torch.concat([labels,node_classes],dim=0)
            sampling_src_idx = torch.concat([sampling_src_idx,src_indices],dim=0)
            
    virtual_feat = x0.squeeze(1)
    v_information = {"feat":torch.detach(virtual_feat),"label":torch.detach(labels.to(y.dtype))}

    torch.cuda.empty_cache()
    return v_information,sampling_src_idx

def isolate_sampling_test(x,y,class_,n_cls,diffusion_model,teacher,temperature, aug_size , padding,guidance=7.5, hard_factor = 0.5, is_hard_sample = True,device="cuda:0"):
    n_emb = x.shape[1]
    print("Generating {} samples for class{}".format(int(aug_size),class_))
    node_classes = torch.full([int(aug_size)],fill_value=class_,device=device)
    # filter hard sample
    hard_labels = y
    class_mask = (hard_labels == class_)
    soft_labels = teacher.softmax_with_temperature(x,temperature)
    nodes_confidence = 1-torch.index_select(soft_labels[class_mask], dim = 1, index=torch.tensor(class_,device=device)).view(-1)
    if is_hard_sample:
        _,indices = torch.topk(nodes_confidence,sum(class_mask)//2)
    else:
        indices = torch.ones((sum(class_mask),),dtype=torch.bool,device=device)
    hard_samples = soft_labels[class_mask][indices]
    mean_confidence = torch.mean(hard_samples,dim=0)
    variance_confidence = mean_confidence / 100
    # print("mean ", mean_confidence)
    # print("var ", variance_confidence)
    # assume that confidence follow Beta districution
    alpha = mean_confidence * (mean_confidence * (1 - mean_confidence) / variance_confidence - 1)
    beta_param = (1 - mean_confidence) * (mean_confidence * (1 - mean_confidence) / variance_confidence - 1)
    # print("alpha ", alpha)
    # print("beta ", beta_param)
    confidence_sample = torch.distributions.Beta(alpha, beta_param).sample((aug_size,)).to(device)
    confidence_guidance = confidence_sample / torch.sum(confidence_sample,dim=1,keepdim=True)
    overall_guidance = confidence_guidance + F.one_hot(node_classes,n_cls) * hard_factor
    x_t = torch.randn([aug_size,1,n_emb]).to(device)
    x0_v,_ = diffusion_model.sampling([None,None],classifier_scale_mode=0,guidance_scale=guidance
                                        ,x_t=x_t,y=overall_guidance,padding=padding,save_frames=False,device=device)
    print("generated node feature quility:")
    logits = teacher(torch.detach(x0_v).squeeze(1))
    preds = logits.argmax(1)
    print(preds)
    

@torch.no_grad()
def add_new_nodes(data, feats, labels, edge_predicter, edge_index, data_train_mask, train_edge_mask, device='cuda:0'):
    rand_factor = 4
    new_node_num = feats.size(0)
    data = copy.deepcopy(data)
    ori_num = data.num_nodes
    total_node = ori_num + new_node_num
    data_train_mask = torch.cat([data_train_mask, torch.ones(new_node_num, dtype=torch.bool, device=device)])

   # Calculate the degree distribution of existing nodes
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col, dim=0, dim_size=data.num_nodes)
    # update graph data
    data.x = torch.cat([data.x, feats.to(device)], dim=0)
    data.y = torch.cat([data.y, labels.to(device)], dim=0)
    with torch.no_grad():
        edge_predicter.eval()
        emb = edge_predicter.linear(data.x)     
        new_edges_src = []
        new_edges_dst = []
        for new_idx in range(ori_num, total_node):
            c_degree = degree[data.y[:ori_num] == data.y[new_idx]]
            degree_dist = torch.bincount(c_degree, minlength=c_degree.max().item() + 1)
            while True:
                d = torch.multinomial(degree_dist.to(dtype=torch.float32), 1).item()  # Sample degree value
                if d != 0: break
            # d = 4
            # compute edge scores between new nodes and exist nodes
            scores = (emb[new_idx] * emb[:ori_num]).sum(dim=-1)
            top_n_d_nodes = torch.topk(scores, rand_factor * d).indices
            # Randomly select d nodes from top_n_d_nodes
            sampled_nodes = torch.multinomial(torch.ones(rand_factor * d, device=device), d)
            # update new edge
            new_edges_src.append(torch.full((d,), new_idx, dtype=torch.long, device=device))
            new_edges_dst.append(top_n_d_nodes[sampled_nodes])

        new_edges_src = torch.cat(new_edges_src, dim=0)
        new_edges_dst = torch.cat(new_edges_dst, dim=0)
        # add new edges to edge_index
        edge_index = torch.cat([edge_index, torch.stack([new_edges_src, new_edges_dst], dim=0)], dim=1)
        # update train_edge_mask, which contain new edges
        new_edge_train_mask = torch.ones(new_edges_src.size(0), dtype=torch.bool, device=device)
        train_edge_mask = torch.cat([train_edge_mask, new_edge_train_mask])

    return data, edge_index, data_train_mask, train_edge_mask

@torch.no_grad()
def neis_mixup(data, feats, labels, edge_index, data_train_mask, train_edge_mask, device='cuda:0'):
    rand_factor = 4
    new_node_num = feats.size(0)
    data = copy.deepcopy(data)
    ori_num = data.num_nodes
    num_edge = data.num_edges
    total_node = ori_num + new_node_num
    data_train_mask = torch.cat([data_train_mask, torch.ones(new_node_num, dtype=torch.bool, device=device)])

   # Calculate the degree distribution of existing nodes
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col, dim=0, dim_size=data.num_nodes)
    # update graph data
    sim_matrix = torch.mm(feats,data.x.t()).clamp(min=1e-4)
    _, v_indx = sim_matrix.topk(k=2*rand_factor, dim=-1)
    data.x = torch.cat([data.x, feats.to(device)], dim=0)
    data.y = torch.cat([data.y, labels.to(device)], dim=0)
    new_edges = []
    
    for new_node_idx, nodes in enumerate(v_indx):  # Iterate over each new node's top similar nodes
        # Randomly select 2 nodes from the 2*r most similar nodes
        start_nodes = nodes[random.sample(range(len(nodes)), 2)]
        
        neighbors = set()
        
        for start_node in start_nodes:
            # Get the neighbors of the selected node from edge_index
            neighbor_mask = edge_index[0] == start_node
            neighbor_nodes = edge_index[1][neighbor_mask]
            neighbors.update(neighbor_nodes.tolist())
        
        # Randomly select half of the neighbors
        neighbors = list(neighbors)
        sampled_neighbors = random.sample(neighbors, len(neighbors) // 2)
        
        # Add edges between the new node and the sampled neighbors
        new_edges.extend([(ori_num + new_node_idx, neighbor) for neighbor in sampled_neighbors])
        
    # Convert new_edges to torch tensors and update edge_index
    if new_edges:
        new_edges = torch.tensor(new_edges, dtype=torch.long, device=device).t()
        edge_index = torch.cat([edge_index, new_edges], dim=1)
    
        # Update train_edge_mask to account for new edges
        new_edge_mask = torch.ones(new_edges.size(1), dtype=torch.bool, device=device)  # All new edges have mask=True
        train_edge_mask = torch.cat([train_edge_mask, new_edge_mask], dim=0)
    
    return data, edge_index, data_train_mask, train_edge_mask

@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx,
        neighbor_dist_list, train_node_mask=None):
    """
    Neighbor Sampling - Mix adjacent node distribution and samples neighbors from it
    Input:
        total_node:         # of nodes; scalar
        edge_index:         Edge index; [2, # of edges]
        sampling_src_idx:   Source node index for augmented nodes; [# of augmented nodes]
        sampling_dst_idx:   Target node index for augmented nodes; [# of augmented nodes]
        neighbor_dist_list: Adjacent node distribution of whole nodes; [# of nodes, # of nodes]
        prev_out:           Model prediction of the previous step; [# of nodes, n_cls]
        train_node_mask:    Mask for not removed nodes; [# of nodes]
    Output:
        new_edge_index:     original edge index + sampled edge index
        dist_kl:            kl divergence of target nodes from source nodes; [# of sampling nodes, 1]
    """
    ## Exception Handling ##
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    
    # Find the nearest nodes and mix target pool
    mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    # Compute degree
    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)
    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    # Sample degree for augmented nodes
    prob = degree_dist.unsqueeze(dim=0).repeat(len(sampling_src_idx),1)
    aug_degree = torch.multinomial(prob, 1).to(device).squeeze(dim=1) # (m)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    # Sample neighbors
    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)
    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index