import torch 
import torch.nn as nn
import math

class LayerNormalisation(nn.Module):
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))     #learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))     #learnable parameter

    def forward(self, x):
        # x : (batch, seq_len, hidden_size)
        # keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True)     # (batch, seq_len, 1)
        # keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True)       # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x : (batch, seq_len, d_model)
        # (batch, seq_len, d_model) * (d_model, d_ff) -> (batch, seq_len, d_ff) * (d_ff, d_model) -> (batch, seq_len, d_model) 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # using torch's given embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # self.pe is a buffer, so it already carries requires_grad=False
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features:int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention_scores = None

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        # q, k, v : (batch, h, seq_len, d_k)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            # mask convention: 1/True == keep, 0/False == hide
            attention_scores = attention_scores.masked_fill(mask == 0, torch.finfo(attention_scores.dtype).min)
        p_attn = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


    def forward(self, query, key, value, mask=None):
        # query, key, value : (batch, seq_len, d_model)
        batch_size = query.size(0)
        # linear projection and split into h heads
        query = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
        key = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)      # (batch, h, seq_len, d_k)
        value = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask=mask, dropout=self.dropout)
        # concatenate heads and put through final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)
        return self.w_o(x)  # (batch, seq_len, d_model)


class EncoderLayer(nn.Module):
    def __init__(self, features:int, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(features)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention: MultiHeadAttentionBlock, cross_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # src : (batch, seq_len_src) -> (batch, seq_len_src, d_model)
        src = self.src_pos(self.src_embedding(src))
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # tgt : (batch, seq_len_tgt) -> (batch, seq_len_tgt, d_model)
        tgt = self.tgt_pos(self.tgt_embedding(tgt))
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.projection_layer(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src_mask : (batch, 1, 1, seq_len_src)
        # tgt_mask : (batch, 1, seq_len_tgt, seq_len_tgt)
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)


def build_transformer(vocab_size_src:int, vocab_size_tgt:int, max_seq_len:int, d_model:int=512, num_layers:int=6, num_heads:int=8, d_ff:int=2048, dropout:float=0.1) -> Transformer:
    # create embeddings layers
    src_embedding = InputEmbedding(d_model, vocab_size_src)
    tgt_embedding = InputEmbedding(d_model, vocab_size_tgt)

    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, max_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderLayer(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder Blocks
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # nn.ModuleList so the layers' parameters are actually registered on the model
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, vocab_size_tgt)

    # create the transformer model
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
