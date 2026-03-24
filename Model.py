from Layers import *

class AD_GAT(nn.Module):
    """Attribute-Driven Graph Attention Network (AD-GAT) for stock movement prediction.

    The model jointly processes **numerical market features** (price/volume indicators)
    and **textual sentiment features** (Loughran-McDonald sentiment scores derived from
    financial news).  Two fusion strategies are supported via the ``t_mix`` flag:

    t_mix == 0 – Simple concatenation
        The sentiment vector and the market vector are concatenated along the feature
        dimension before being fed into the GRUs.

    t_mix == 1 – Bilinear tensor fusion (default)
        The ``Graph_Tensor`` module first fuses sentiment and market vectors using a
        learnable per-stock bilinear tensor interaction:
            fused = tanh( news^T · T · market  +  [news; market] · W  +  b )
        This allows the model to capture multiplicative cross-modal dependencies
        (e.g. *which* sentiment dimensions amplify or dampen *which* market signals).

    After fusion the representation is processed by two parallel GRUs:
    - ``GRUs_r`` produces an embedding used to *infer dynamic stock relations*.
    - ``GRUs_s`` produces an embedding used for *attribute-mattered propagation*.

    Both GRU outputs carry the fused sentiment+market information into the graph
    attention layers, so sentiment data influences both *who attends to whom* and
    *what is propagated* across the stock graph.

    Args:
        num_stock (int):  Number of stocks (graph nodes).
        d_market  (int):  Dimension of the numerical market feature vector.
        d_news    (int):  Dimension of the textual sentiment feature vector.
        d_hidden  (int):  Hidden dimension used by Graph_Tensor fusion.
        hidn_rnn  (int):  Hidden dimension of the GRU cells.
        heads_att (int):  Number of graph attention heads.
        hidn_att  (int):  Hidden dimension of each attention head.
        dropout   (float): Dropout probability.
        alpha     (float): LeakyReLU negative slope for attention.
        t_mix     (int):  0 = concat fusion, 1 = bilinear tensor fusion.
        infer     (int):  Whether to infer dynamic relations (currently unused flag).
        relation_static (int): 1 if a static relation matrix is provided, else 0.
    """

    def __init__(self, num_stock, d_market, d_news, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1, infer = 1, relation_static = 0):
        super(AD_GAT, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        if self.t_mix == 0: # concat: simply concatenate sentiment and market vectors
            self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
        elif self.t_mix == 1: # bilinear tensor fusion of sentiment and market vectors
             self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_news)
             self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
             self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.attentions = [
            Graph_Attention(hidn_rnn, hidn_att, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
            in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.X2Os = Graph_Linear(num_stock, heads_att * hidn_att  + hidn_rnn , 2, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def get_relation(self,x_numerical, x_textual, relation_static = None):
        """Infer dynamic pairwise stock relations from fused sentiment+market data.

        The sentiment features (x_textual) are fused with market features (x_numerical)
        via Graph_Tensor, then encoded by GRUs_r to obtain a temporal embedding.
        Graph attention heads compute a relation score matrix from this embedding.
        """
        x_r = self.tensor(x_numerical, x_textual)
        x_r = self.GRUs_r(x_r)
        relation = torch.stack([att.get_relation(x_r, relation_static=relation_static) for att in self.attentions])
        # relation_mean = torch.mean(abs_relation,dim = 1)
        return relation

    def get_gate(self,x_numerical,x_textual):
        """Compute attribute-based propagation gates from fused sentiment+market data."""
        x_s = self.tensor(x_numerical, x_textual)
        x_s = self.GRUs_s(x_s)
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, x_market, x_news, relation_static = None):
        """Forward pass.

        Sentiment data usage pipeline
        ------------------------------
        1. **Fusion** (t_mix == 1, default):
           ``Graph_Tensor`` fuses ``x_market`` (numerical) and ``x_news`` (sentiment)
           into a single representation via bilinear tensor interaction:
               fused = tanh( news^T · T · market + [news; market] · W + b )

           With t_mix == 0, market and sentiment vectors are simply concatenated.

        2. **Temporal encoding**:
           Two separate GRUs process the fused sequence:
           - ``GRUs_r`` → temporal embedding for *relation inference*.
           - ``GRUs_s`` → temporal embedding for *node feature propagation*.

        3. **Relation inference** (uses sentiment via GRUs_r output ``x_r``):
           Each attention head computes a dynamic N×N relation matrix from ``x_r``.
           Sentiment signals therefore influence *which stocks attend to which*.

        4. **Attribute-mattered propagation** (uses sentiment via GRUs_s output ``x_s``):
           Each attention head applies a per-pair gate to ``x_s`` before aggregation,
           so sentiment also modulates *what information is propagated* across edges.

        5. **Prediction**:
           The aggregated representations from all heads are concatenated with ``x_s``
           and projected to a 2-class log-softmax output (up/down movement).

        Args:
            x_market (Tensor): shape (T, num_stocks, d_market) – market features.
            x_news   (Tensor): shape (T, num_stocks, d_news)   – sentiment features.
            relation_static: optional static relation matrix for hybrid attention.

        Returns:
            Tensor of shape (num_stocks, 2) – log-softmax class probabilities.
        """
        # --- Step 1: fuse sentiment (x_news) and market (x_market) features ---
        if self.t_mix == 0:  # concat
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:  # bilinear tensor fusion
            x_s = self.tensor(x_market, x_news)
            x_r = self.tensor(x_market, x_news)

        # --- Step 2: temporal encoding via separate GRUs ---
        # GRUs_r encodes the fused sequence for dynamic relation inference
        # GRUs_s encodes the fused sequence for attribute-mattered propagation
        x_r = self.GRUs_r(x_r)
        x_s = self.GRUs_s(x_s)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)

        # --- Steps 3 & 4: graph attention (relation inference + gated propagation) ---
        x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        # --- Step 5: residual skip connection from x_s, then linear projection ---
        x = torch.cat([x, x_s], dim=1)
        x = F.elu(self.X2Os(x))
        output = F.log_softmax(x, dim=1)
        return output