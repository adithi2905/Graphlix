import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedNGCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        self.W = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)])
        self.W_self = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)])
        self.attn = nn.ModuleList([nn.Linear(emb_dim * 2, 1) for _ in range(num_layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        emb = ego_embeddings

        for k in range(self.num_layers):
            side_embeddings = torch.sparse.mm(adj, emb)
            attn_input = torch.cat([emb, side_embeddings], dim=1)
            attn_weight = torch.sigmoid(self.attn[k](attn_input))
            neighbor_output = attn_weight * side_embeddings

            sum_embed = self.W[k](neighbor_output)
            bi_embed = self.W_self[k](emb * side_embeddings)
            layer_output = sum_embed + bi_embed

            layer_output = self.leaky_relu(layer_output)
            layer_output = self.bn[k](layer_output)
            layer_output = self.dropout(layer_output)
            emb = F.normalize(layer_output, p=2, dim=1)

            all_embeddings.append(emb)

        final_emb = torch.cat(all_embeddings, dim=1)
        user_embs = final_emb[:self.num_users]
        item_embs = final_emb[self.num_users:]
        return user_embs, item_embs

    def bpr_loss(self, users, pos_items, neg_items, adj, reg_lambda=1e-4):
        user_embs, item_embs = self.forward(adj)
        u = user_embs[users]
        pos = item_embs[pos_items]
        neg = item_embs[neg_items]
        pos_scores = torch.sum(u * pos, dim=1)
        neg_scores = torch.sum(u * neg, dim=1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = (1/2)*(u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / float(len(users))
        return loss + reg_lambda * reg_loss

    def get_scores(self, adj):
        user_embs, item_embs = self.forward(adj)
        return torch.matmul(user_embs, item_embs.T)
