import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads:int = 1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == embed_dim), "embedding must be divided by number of heads!"
        
        # Linear transformations for queries, keys, and values
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        
        # Linear transformation for the output of attention heads
        self.linear_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, n_channels, embed_dim = x.size()
        
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)
        Q = Q.view(batch_size, n_channels, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_channels, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_channels, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum('bhqd, bhkd -> bhqk', Q, K) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.einsum('bhqk, bhkd -> bhqd', attention_weights, V)
        # Reshape attended_values to (batch_size, n_channel, embed_dim)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, n_channels, embed_dim)
        output = self.linear_out(attended_values)
        output += x # resconnect
        
        return output
