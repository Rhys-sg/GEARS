import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import SGConv

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)
    
# class AE(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         """
#         Autoencoder model
#         :param input_dim: dimension of the input data
#         :param hidden_dim: dimension of the hidden representation
#         """
#         super().__init__()
#         # Encoder: compress to hidden_size
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#         # Decoder: reconstruct back to input_dim
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, input_dim)
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         recon = self.decoder(z)
#         return z, recon



class GEARS_Model(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb_go = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        self.pert_emb_BioGRID = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        self.pert_emb_comb_trans = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        # self.pert_autoencoder = AE(input_dim=hidden_size*2, hidden_dim=hidden_size)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### First perturbation GNN weights
        self.G_sim_go = args['G_go'].to(args['device'])
        self.G_sim_weight_go = args['G_go_weight'].to(args['device'])

        # Second perturbation GNN weights
        self.G_sim_BioGRID = args['G_BioGRID'].to(args['device'])
        self.G_sim_weight_BioGRID = args['G_BioGRID_weight'].to(args['device'])

        # Separate GNN stacks for each embedding
        self.sim_layers1 = torch.nn.ModuleList()
        self.sim_layers2 = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.sim_layers1.append(SGConv(hidden_size, hidden_size, 1))
            self.sim_layers2.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T     

            pert_emb_go = self.pert_emb_go(torch.arange(self.num_perts, device=self.args['device']))
            pert_emb_BioGRID = self.pert_emb_BioGRID(torch.arange(self.num_perts, device=self.args['device']))

            # Process each pert embedding through its own GNN
            for idx, layer in enumerate(self.sim_layers1):
                pert_emb_go = layer(pert_emb_go, self.G_sim_go, self.G_sim_weight_go)
                if idx < self.num_layers - 1:
                    pert_emb_go = pert_emb_go.relu()

            for idx, layer in enumerate(self.sim_layers2):
                pert_emb_BioGRID = layer(pert_emb_BioGRID, self.G_sim_BioGRID, self.G_sim_weight_BioGRID)
                if idx < self.num_layers - 1:
                    pert_emb_BioGRID = pert_emb_BioGRID.relu()
            
        
            # Add the two GNN outputs, then apply nonlinearity using an MLP
            pert_global_emb = pert_emb_go + pert_emb_BioGRID
            pert_global_emb = self.pert_emb_comb_trans(pert_global_emb)

            # # Concatenate the two GNN outputs, then project to hidden_size using an autoencoder
            # pert_global_emb_stacked = torch.cat([pert_emb_go, pert_emb_BioGRID], dim=-1)
            # pert_global_emb, recon = self.pert_autoencoder(pert_global_emb_stacked)

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)
        
