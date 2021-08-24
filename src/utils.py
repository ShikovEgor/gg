import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from graph_tool.all import *
from torch_geometric.utils import k_hop_subgraph


# def create_nx_graph(edge_list, lbs):
#     v,inv = torch.unique(edge_list, return_inverse=True)

#     colours = {0:'red', 1:'blue', 2:'green',3:'yellow', 4:'grey', 5:'orange',6:'black'}
#     labs = [colours.get(x,'black') for x in lbs[v].numpy()]
    
#     P=nx.Graph()
#     P.add_edges_from(inv.numpy().T)
#     P = P.to_undirected()

#     largest_cc = max(nx.connected_components(P), key=len)
#     G_conn = P.subgraph(largest_cc).copy()
    
#     lbs_fin =[]
#     for i in range((len(labs))):
#         if i in largest_cc:
#             lbs_fin.append(labs[i])
    
#     return G_conn, lbs_fin
class NodeSelector:
    def __init__(self, lims, vects):
        self.base_lims = torch.tensor(lims)
        self.vects = vects
        self.update_vects(vects)
    
    def update_vects(self, vects):
        self.c_sum = torch.cumsum(vects, dim=0)
        
    def cum_sum_diff(self, ind_low, ind_high):
        msk0 = ind_low != 0
        csum_diff = self.c_sum[ind_high].clone()
#         msk0 = ~( (ind_low == 0) | (ind_low==ind_high) )
        
        csum_diff[msk0] -= self.c_sum[ind_low-1][msk0]
        return csum_diff    
        
    def select_nodes(self, n):
        self.limits = self.base_lims.repeat(n,1)
        return self.__sel()
        
    def __sel(self):
#         msk0122 =torch.all(self.limits == torch.tensor([0,1,2,2]),dim=1) 
#         print(self.limits)
        
#         dx = self.limits[0,1]-self.limits[0,0] 
#         dy = self.limits[0,1]-self.limits[0,0]
        
        if (self.limits[0,1]-self.limits[0,0]) <2: # 2x2 or 1x2 or 2x1 matrix
            quadrants = [torch.stack((self.limits[:,i],self.limits[:,j]), dim=1) for i in range(2) for j in range(2,4)]
#             print(quadrants[0].shape)
#             print('quadrants')
#             print([q[msk0122,:] for q in quadrants])
    
            prob_quad = torch.stack([(self.vects[q[:,0]]*self.vects[q[:,1]]).sum(1) for q in quadrants], dim=1)
            quadrants = torch.stack(quadrants,dim=2)
            
            
            diag_mask = self.limits[:,0]==self.limits[:,2]
            x_mask = self.limits[:,0]==self.limits[:,1]
            y_mask = self.limits[:,2]==self.limits[:,3]
#             fin_mask = diag_mask |  y_mask
            
            prob_quad[diag_mask, 0] = 0
            prob_quad[diag_mask, 2:] = 0            
            
            prob_quad[y_mask, 0] = 0
            prob_quad[y_mask, 2] = 0
            
            prob_quad[x_mask, :] = 1 # for 1x1 matrix
        else:
            mid_r = (self.limits[:,0] + self.limits[:,1]) // 2 
            mid_c = (self.limits[:,2] + self.limits[:,3]) // 2    
            quadr_lims = ((self.limits[:,0], mid_r), (mid_r+1, self.limits[:,1]),
                          (self.limits[:,2], mid_c), (mid_c+1, self.limits[:,3]))
            
            
            w_slice = [self.cum_sum_diff(l_low, l_high) for (l_low,l_high) in quadr_lims]
            quadr_lims = [torch.stack((q[0],q[1]),dim=1) for q in quadr_lims]
#             print(quadr_lims)
        
            quadrants = torch.stack([torch.cat((quadr_lims[i], quadr_lims[j]), dim=1) for i in range(2) for j in range(2,4)],dim=2)
            prob_quad = torch.stack([(w_slice[i]*w_slice[j]).sum(1) for i in range(2) for j in range(2,4)], dim=1)
            
#             print(quadrants)
            
            # delete diagonal elements' contribution
#             print('1111111111111')
#             print(prob_quad)
            diag_mask = self.limits[:,0]==self.limits[:,2]
            prob_quad[diag_mask, 0] -= (quadr_lims[0][diag_mask,1]-quadr_lims[0][diag_mask,0]+1)
            prob_quad[diag_mask, 3] -= (quadr_lims[3][diag_mask,1]-quadr_lims[3][diag_mask,0]+1)
        
#         diag_mask = self.limits[:,0]==self.limits[:,2]
        prob_quad[diag_mask, 2] = 0 # generate only U matrixes
        prob_quad[diag_mask, 0] /= 2
        prob_quad[diag_mask, 3] /= 2
#         print('-----------------prob_quad-----------------')
#         print(prob_quad)
#         print(torch.all(prob_quad>0, dim=1).shape, torch.squeeze(torch.multinomial(prob_quad, 1)).shape, torch.zeros())
            
#         ind_quad = torch.where(torch.any(prob_quad>0, dim=1), 
#                                torch.squeeze(torch.multinomial(prob_quad, 1)), 
#                                torch.zeros(self.limits.shape[0], dtype=torch.long))
        ind_quad = torch.squeeze(torch.multinomial(torch.clamp(prob_quad, min=0), 1))  
        if self.limits[0,1]-self.limits[0,0] <2:
            return quadrants[torch.arange(self.limits.shape[0]),:, ind_quad] 
        
        self.limits = quadrants[torch.arange(self.limits.shape[0]),:,ind_quad]
        return self.__sel()         


        
def get_errors(rz, thr):
    n_e_pred_sum, n_e_true_sum = 0,0
    for pred_e, targ_e, eps_0 in rz:
        pr = (pred_e-thr)>0
        tp = pr[targ_e.to(dtype=torch.bool)].sum()
        fp = pr[~targ_e.to(dtype=torch.bool)].sum()
        fp = fp/eps_0
        n_e_pred_sum += tp + fp
        n_e_true_sum += targ_e.sum()   
    return torch.abs(n_e_pred_sum-n_e_true_sum)

def sel_start_node_old(nds):
    _, degs = torch.unique(data.edge_index, return_counts = True)
    deginds = torch.argsort(degs[nds], descending=True)
    sel_ind = nds[deginds[int(deginds.shape[0]/2)]]
    return sel_ind

def sel_start_node(nds, data):
    gr = Graph()
    gr.add_edge_list(data.edge_index.numpy().T)
    pr = pagerank(gr).get_array()[nds.numpy()]
    thr  = np.quantile(pr, 0.8) #np.median(pr)
    central_nodes = np.where(pr>thr)[0]  
    sel_ind = central_nodes[np.argmin(pr[central_nodes])]
    return nds[sel_ind]

def init_graph(node_set, data):
    stt = sel_start_node(node_set, data)
    start_nds, ei0, _, _ = k_hop_subgraph([stt], 2, data.edge_index,relabel_nodes=True)
    in_start = (node_set[..., None] == start_nds).any(-1)
    node_set = torch.cat((start_nds, node_set[~in_start]))
    return start_nds, ei0

def relabel(graph, nodes):
    old_inds = torch.zeros(nodes.shape[0], dtype=torch.long)
    new_inds = torch.unique(graph)
    old_inds[new_inds] = torch.arange(new_inds.shape[0])
    return old_inds[graph], new_inds

def draw_deg_distr(gr_gen, gr_true):
    _, degsnew = torch.unique(gr_gen, return_counts = True)
    _, degs = torch.unique(gr_true, return_counts = True)

    plt.figure(figsize=(8,6))
    plt.hist(np.log(degs*2).numpy(), density=True, histtype='step', lw=3, log=False, range=(0,5), label = 'True') 
    plt.hist(np.log(degsnew).numpy(), density=True, histtype='step', lw=3, log=False, range=(0,5), label = 'Generated')   

    plt.legend()
    plt.show()
    

def create_gt_graph(edge_list, lbs, get_largest = True):
    c,r = edge_list
    edge_list = edge_list[:, c>r]
    colours = {0:'red', 1:'blue', 2:'green',3:'yellow', 4:'grey', 5:'orange',6:'black'}
#     labs = [colours.get(x,'black') for x in lbs.numpy()]
    labs = lbs.numpy()
    G_conn = Graph()
    G_conn.add_edge_list(edge_list.numpy().T)
    lbs_fin = G_conn.new_vertex_property("int")   
    lbs_fin.set_2d_array(lbs.numpy())
    if get_largest:
        G_conn = extract_largest_component(G_conn, prune=False, directed = False)
        return G_conn, [labs[i] for i in G_conn.get_vertices()], lbs_fin
    return G_conn, labs, lbs_fin

def create_nx_graph(edge_list, lbs, get_largest = True):
    v,inv = torch.unique(edge_list, return_inverse=True)
#     print(v)
    colours = {0:'red', 1:'blue', 2:'green',3:'yellow', 4:'grey', 5:'orange',6:'black'}
    labs = [colours.get(x,'black') for x in lbs[v].numpy()]
    
    P=nx.Graph()
    P.add_edges_from(inv.numpy().T)
    P = P.to_undirected()

    G_conn = P
    lbs_fin = labs
    if get_largest:
        largest_cc = max(nx.connected_components(P), key=len)
        G_conn = P.subgraph(largest_cc).copy()
        lbs_fin =[]
        for i in range((len(labs))):
            if i in largest_cc:
                lbs_fin.append(labs[i])

    return G_conn, lbs_fin