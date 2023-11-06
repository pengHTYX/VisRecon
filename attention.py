import torch
from torch import nn
import torch.nn.functional as F
from icecream import ic


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        """
            Scaled Dot-Product Attention
            out = softmax(QK^T/temperature)V
        :param q: (bs, lenq, d_k)
        :param k: (bs, lenv, d_k)
        :param v: (bs, lenv, d_v)
        :param mask: None
        :return:
        """
        # print(q.shape,v.shape)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # print(attn.shape)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # print(attn.shape)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        # self.bn = nn.BatchNorm1d(d_model)
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        """
        :param q: (bs, lenq, d_model)
        :param k: (bs, lenv, d_model)
        :param v: (bs, lenv, d_model)
        :param mask:
        :return: (bs, lenq, d_model)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        bs, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: bs x lq x (n*dv)
        # Separate different heads: bs x lq x n x dv
        q = self.w_qs(q).view(bs, len_q, n_head, d_k)
        k = self.w_ks(k).view(bs, len_k, n_head, d_k)
        v = self.w_vs(v).view(bs, len_v, n_head, d_v)

        # Transpose for attention dot product: bs x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: bs x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b sx lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(bs, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        # q = self.bn(q.permute((0, 2, 1))).permute((0, 2, 1))
        # q = self.layer_norm(q)
        if self.norm:
            q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1, norm=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)    # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)    # position-wise
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.bn = nn.BatchNorm1d(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        if self.norm:
            # x = self.bn(x.permute((0, 2, 1))).permute((0, 2, 1))
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """
        multi-attention + position feed forward
    """

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1,
                 norm=False):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head,
                                           d_model,
                                           d_k,
                                           d_v,
                                           dropout=dropout,
                                           norm=norm)
        self.pos_ffn = PositionwiseFeedForward(d_model,
                                               d_inner,
                                               dropout=dropout,
                                               norm=norm)

    def forward(self, enc_input):
        # print(type(enc_input))
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input,
                                                 enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, 0


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 dropout=0.1,
                 norm=False,
                 n_position=200):

        super().__init__()

        # self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model,
                         d_inner,
                         n_head,
                         d_k,
                         d_v,
                         dropout=dropout,
                         norm=norm) for _ in range(n_layers)
        ])
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.bn = nn.BatchNorm1d(d_model)

    def forward(self, feats_embedding, return_attns=False):
        """
        :param feats_embedding: (bs, num_views, dim)
        :param return_attns:
        :return:
        """
        enc_slf_attn_list = []
        # enc_output = self.dropout(self.position_enc(feats_embedding))
        enc_output = feats_embedding
        if self.norm:
            enc_output = self.layer_norm(enc_output)
        # enc_output = feats_embedding
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


def sdf_fusion(x, fusion_net, view_num=3):
    '''
    :param x: [batch*view_num,dim,point_num,1], dim=32+2+3+3
    :return: [batch*view_num,dim,point_num,1], dim=32+2+3+3
    '''

    x = x.squeeze(-1)
    bs_v, dim, pts_num = x.shape

    x = x.reshape(-1, view_num, dim, pts_num)
    x = x.permute(0, 3, 1, 2).reshape(-1, view_num,
                                      dim)    # batch*point_num,view_num.dim
    # print(x.shape)

    # feature_fusion = Encoder(n_head=n_head,n_layers=n_layers,d_k=d_k,d_v=d_v,d_model=d_model,d_inner=d_inner).to(device)

    x_fusion = fusion_net(x)    #batch*point_num,view_num, d_model
    # print(x_fusion.shape)
    x_fusion = x_fusion.reshape(-1, pts_num, view_num,
                                dim).permute(0, 2, 3, 1).unsqueeze(-1)
    x_fusion = x_fusion.reshape(-1, dim, pts_num, 1)
    # print(x_fusion.shape)
    return x_fusion


if __name__ == '__main__':
    x = torch.randn(3 * 3, 40, 5000, 1).cuda()
    fusion = Encoder(n_head=8,
                     n_layers=2,
                     d_k=16,
                     d_v=16,
                     d_model=40,
                     d_inner=64).to('cuda')
    sdf_fusion(x, fusion)

    attn = torch.randn(1, 1, 3, 3)
    print(attn)
    attn = F.softmax(attn, dim=-1)
    print(attn)
