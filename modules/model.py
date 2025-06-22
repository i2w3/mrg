import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

class ImageFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()
        model = timm.create_model(backbone, pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # 去掉分类头
        self.feature_dim = model.num_features

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x).squeeze(-1).squeeze(-1)  # [B*T, D]
        return feats.view(B, T, -1)  # [B, T, D]

class Sampler(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, feats):  # [B, T, D]
        scores = self.linear(feats).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=-1)
        return weights  # [B, T]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ReportGenerator(nn.Module):
    def __init__(self, feature_dim, vocab_size, d_model=512, num_layers=3):
        super().__init__()
        self.img_feat_proj = nn.Linear(feature_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True  # ✅ 关键设置：batch_first=True
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, img_feats, tgt_seq):
        """
        img_feats: [B, T_img, feature_dim]
        tgt_seq:   [B, T_tgt]  —— token ids
        """
        B, T_img, _ = img_feats.shape
        B, T_tgt = tgt_seq.shape

        # 图像特征 → 投影 & 位置编码
        enc = self.img_feat_proj(img_feats)       # [B, T_img, d_model]
        enc = self.pos_encoder(enc)

        # 文本 token → 嵌入 & 位置编码
        tgt_emb = self.token_embedding(tgt_seq)   # [B, T_tgt, d_model]
        tgt_emb = self.pos_encoder(tgt_emb)

        # Transformer 编-解码
        output = self.transformer(
            src=enc,
            tgt=tgt_emb,
            tgt_mask=self._generate_square_subsequent_mask(T_tgt, device=tgt_seq.device)
        )  # [B, T_tgt, d_model]

        logits = self.fc_out(output)  # [B, T_tgt, vocab_size]
        return logits

    def _generate_square_subsequent_mask(self, sz, device=None):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)


class JointModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.extractor = ImageFeatureExtractor()
        self.sampler = Sampler(self.extractor.feature_dim)
        self.generator = ReportGenerator(self.extractor.feature_dim, vocab_size)

    def forward(self, images, tgt_seq):
        feats = self.extractor(images)  # [B, T, D]
        weights = self.sampler(feats)  # [B, T]
        weighted_feats = (feats * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)  # [B, 1, D]
        return self.generator(weighted_feats, tgt_seq), weights
