from torch_geometric.data import InMemoryDataset
import pickle
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_undirected
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from typing import List, Optional, Set, get_type_hints
from torch_scatter import gather_csr, scatter, segment_csr

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
import torch
import numpy as np
from collections import defaultdict


def data_gen_e_aug(train_loader, slices, batch_size = 4, step = 1):
    for ib, data in enumerate(train_loader):
#         r,c = data.edge_index
#         print(data.edge_index)
#         print(data.edge_index[:,r>c].shape)
        e_ptr = slices[ib*batch_size:(ib+1)*batch_size+1]
        e_ptr = e_ptr - e_ptr[0]
        szs = e_ptr[1:]-e_ptr[:-1]
        e_ind_start = e_ptr[1:]-szs.min()+1
        visited_e = torch.full((e_ptr[-1],), False, dtype = torch.bool)
        for i in range(e_ind_start.shape[0]):
            visited_e[e_ptr[:-1][i] :e_ind_start[i]] = True # setting emask True for edges in graph
        edges_num = torch.arange(e_ptr[-1])

        visited_v = torch.full((data.ptr[-1],), False)
        visited_v[torch.unique(data.edge_index[:, visited_e])] = True
        
        vert_ind = []
        last_ind_v = torch.arange(data.ptr[-1])
        last_ind_max = data.ptr[-1].item()        
        ei_dict = defaultdict(set)
        base_edges = set()
        for e in data.edge_index[:, visited_e].T:
            e = e.to(dtype=torch.long)
            ei_dict[e[0].item()].add(e[1].item()) 
            ei_dict[e[1].item()].add(e[0].item()) 
            base_edges.add(e[0].item())
            base_edges.add(e[1].item())
        
        for i in range(data.edge_index.shape[1]): # max number of iterations, usually we stop earlier
            if torch.all(visited_e):
                break           
                  
            e1_mask = (visited_v[data.edge_index[0]] | visited_v[data.edge_index[1]]) & ~visited_e # Source in graph
            nnedges = edges_num[e1_mask]
            e1_ind = []
            for j in range(1, e_ptr.shape[0]):
                mmask = (nnedges < e_ptr[j]) & (nnedges >= e_ptr[j-1])
                e1_ind.append(nnedges[mmask][:step])         
            e1_ind = torch.cat(e1_ind)
            edges_1 = data.edge_index[:, e1_ind]
                        
            for e in edges_1.T: 
                for iv in (True,False):
                    ind = last_ind_v[e[int(iv)]].item()
                    if ind in ei_dict.keys():
                        ind = last_ind_max
                        vert_ind.append(e[int(iv)])
                        last_ind_v[e[int(iv)]] = ind
                        ei_dict[ind] = ei_dict[e[int(iv)].item()]   
                        last_ind_max += 1                        
                    ei_dict[ind].add(last_ind_v[e[int(~iv)]].item())
        
            # selecting source-target when both vertices in graph            
            visited_e[e1_ind] = True  
            visited_v[edges_1.view(-1)] = True
        
        edge_index, e1 = []
        for k,v in ei_dict.items():
            e = torch.tensor(list(v)).view(1,-1)
            edge_index.append(torch.cat((e, torch.full_like(e, k)), dim=0))
            if k not in base_edges:
                e1.append(torch.cat((e, torch.full_like(e, k)), dim=0))

        vert_index = torch.cat((torch.arange(data.ptr[-1]),torch.Tensor(vert_ind))).to(dtype=torch.long)

        yield  data.x[vert_index],\
                torch.cat(edge_index, dim=1)


class RWDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(RWDataset, self).__init__(root, transform, pre_transform)
        self.pth = '/data/egor/graph_generation/graph_generation/data/rw/cora/graphs/'
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.features = torch.load(self.pth+'cora_x.pt')
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['/data/egor/graph_generation/graph_generation/data/rw/cora//cora.dataset']

    def download(self):
        pass
    
    def process(self):
        self.pth = '/data/egor/graph_generation/graph_generation/data/rw/cora/graphs/'

        data_list = []

        edge_list_array = []
        
        for ig in range(14538):
            with open(self.pth + 'graph' + str(ig) + '.dat', 'rb') as f:        
                G = pickle.load(f)
                x, ei = torch.unique(torch.tensor(list(G.edges)).T, return_inverse  = True)
                ei = to_undirected(ei)
#                 data_list.append(Data(x = x, edge_index = ei))
                
                c,r = ei
                data_list.append(Data(x = x, edge_index = ei[:,c<r]))
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])