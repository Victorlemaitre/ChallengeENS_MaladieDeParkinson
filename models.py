import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=config.hidden_dim, num_heads=config.n_head,
                                         dropout=config.dropout_att, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x)[0]


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(config.hidden_dim, config.mul * config.hidden_dim),
            config.act,
            nn.Dropout(config.dropout_MLP),
            nn.Linear(config.mul * config.hidden_dim, config.hidden_dim)

        )

    def forward(self, x):
        out = self.seq(x)
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.MLP_1 = MLP(config)
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.MLP_1(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config, n_features):
        super().__init__()
        self.n_features = n_features
        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.pos_embed = nn.Embedding(config.context_length, config.hidden_dim)
        self.tok_embed = nn.Sequential(
            nn.Linear(n_features, config.mul * config.hidden_dim),
            config.act,
            nn.Linear(config.mul * config.hidden_dim, config.hidden_dim)
        )
        self.final_layer = nn.Linear(config.hidden_dim, config.output_dim)

        self.ln = nn.LayerNorm(config.hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.deviation)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.deviation)

    def forward(self, x):
        """
        Inputs are of shape [batch_size, context_length*nb_features] because we concatenated all rows with the same patient id
        We reshape it into [batch_size, context_length, nb_features] and apply an MLP to obtain the embedded vector [batch_size, context_length, hidden_dimension]
        The final output is of shape [batch_size, context_length, 1]
        """
        x = x.view(-1, self.config.context_length, self.n_features)

        pos_embed = self.pos_embed(torch.arange(self.config.context_length, dtype=torch.int32, device=self.config.device))
        tok_embed = self.tok_embed(x)

        x = tok_embed + pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.final_layer(self.ln(x))
        return x





