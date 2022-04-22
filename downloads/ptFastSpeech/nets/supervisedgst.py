# coding: utf-8
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

class GST(nn.Module):

    def __init__(self, idim, odim, rlayers, gru_unit, num_gst, num_heads):
        super().__init__()

        gru_unit = gru_unit
        num_gst = num_gst
        style_dim = odim
        num_heads = num_heads

        self.encoder = ReferenceEncoder(idim=idim, rlayers=rlayers, gru_unit=gru_unit)
        self.stl = STL(num_gst=num_gst, style_dim=style_dim, gru_unit=gru_unit, num_heads=num_heads)


    def forward(self, inputs, num_emotion, weight=1.0):
        enc_out = self.encoder(inputs) # batch, 128
        style_embed = self.stl(enc_out, num_emotion, weight) # [N, 1, num_units]

        return style_embed.squeeze(1)


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, idim=80, rlayers=[32, 32, 64, 64, 128, 128], gru_unit=128):
        super().__init__()

        self.in_dim = idim
        gru_unit = gru_unit
        convolutions = rlayers
        K = len(convolutions)
        filters = [1] + convolutions
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=convolutions[i]) for i in range(K)])

        out_channels = self.calculate_channels(self.in_dim, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=convolutions[-1] * out_channels,
                          hidden_size=gru_unit,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.in_dim)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, num_gst=10, style_dim=80, gru_unit=80, num_heads=8):

        super().__init__()
        key_dim = style_dim // num_heads
        self.embed = nn.Parameter(torch.FloatTensor(4, num_gst, key_dim))
        self.attention = MultiHeadAttention(query_dim=gru_unit, key_dim=key_dim, num_units=style_dim, num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs, num_emotion, weight=1.0):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = torch.tanh(self.embed) * weight
        keys_emotion = keys[num_emotion]
        style_embed = self.attention(query, keys_emotion)

        return style_embed #* weight # [N, 1, num_units]


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, 1, num_units]
        keys = self.W_key(key)  # [N, num_token, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, 1, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, num_token, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, num_token, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, 1, num_token]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, 1, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, 1, num_units]

        return out