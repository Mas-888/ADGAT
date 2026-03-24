# Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks
Code & Data for the stock prediction model in our paper: Modeling the Momentum Spillover Effect for Stock Prediction via
Attribute-Driven Graph Attention Networks.

## Environment
Python 3.7.6 & Pytorch 1.5.1 

## Run
```sh
$ python main.py --device=0
```
Make sure that the GPU is used to reproduce our experiments.

## How the Model Uses Text Sentiment Data

The model integrates textual sentiment data throughout its pipeline.  The following
describes each stage in detail.

### 1. Data source and preprocessing

Sentiment indicators are derived from Reuters and Bloomberg financial news
(2011-2013, released by Ding et al., EMNLP 2014) using the
**Loughran-McDonald Master Dictionary**.  For each stock on each trading day a
fixed-length vector of sentiment scores is computed (e.g. counts / ratios of
positive, negative, uncertainty, litigious, constraining words).

The pre-computed vectors are stored in `./data/x_textual.pkl` and have shape
`(num_days, num_stocks, d_news)`.

### 2. Bilinear Tensor Fusion – `Graph_Tensor` (Layers.py)

The core mechanism for combining text and market data is the **bilinear tensor
fusion** module `Graph_Tensor`.  For each stock at each time step it computes:

```
fused = tanh( news^T · T · market  +  [news; market] · W  +  b )
```

where:
- `news` (d_hidden) – the sentiment vector projected from d_news-dim to d_hidden-dim  
  via a learnable 1-D convolution `seq_transformation_news`.
- `market` (d_hidden) – the market vector projected from d_market-dim to d_hidden-dim  
  via `seq_transformation_markets`.
- `T` (d_hidden × d_hidden × d_hidden, per stock) – per-stock bilinear tensor that
  captures **multiplicative cross-modal interactions** (which sentiment dimensions
  amplify or dampen which market signals).
- `W` (2·d_hidden × d_hidden, per stock) – linear projection of the concatenated
  [news; market] vector.
- `b` – per-stock bias.

An alternative simpler fusion (`--t_mix 0`) simply concatenates the two vectors
without any learnable interaction term.

### 3. Temporal Encoding – `Graph_GRUModel` (Layers.py)

After fusion, the combined representation is processed by **two separate GRUs**
over the rolling window of `rnn_length` trading days:

| GRU | Purpose |
|-----|---------|
| `GRUs_r` | Encodes the fused sequence for **dynamic stock-relation inference** |
| `GRUs_s` | Encodes the fused sequence for **attribute-mattered propagation** |

Both GRU hidden states carry the blended sentiment+market information forward.

### 4. Relation Inference – `Graph_Attention.get_relation` (Layers.py)

Each attention head uses the `GRUs_r` output (`x_r`) to compute a dynamic N×N
relation/attention matrix.  Because `x_r` encodes fused sentiment features,
**text sentiment affects which stocks attend to which** (i.e. it shapes the graph
topology at each time step).  An optional static relation matrix can be added as
a prior.

### 5. Attribute-Mattered Propagation – `Graph_Attention.forward` (Layers.py)

Each attention head applies a **per-pair gate** to the `GRUs_s` output (`x_s`)
before aggregating neighbour representations.  Since `x_s` also encodes fused
sentiment features, **text sentiment modulates what information is propagated**
across graph edges.

### 6. Prediction

Aggregated representations from all attention heads are concatenated with `x_s`
(skip connection) and linearly projected to a 2-class log-softmax output
(stock-price up / down).

### Pipeline Summary

```
x_textual (d_news)  ──┐
                       ├──► Graph_Tensor (bilinear fusion) ──► fused (d_hidden)
x_numerical (d_market) ┘         │
                                  ├──► GRUs_r ──► x_r ──► relation inference
                                  └──► GRUs_s ──► x_s ──► gated propagation
                                                               │
                                                               └──► X2Os ──► log-softmax
```

## Data
All the preprocessed data can be found at  ./data. 

##### Selected Stock
The selected 198 tickers can be found at ./raw_data/stocks.txt

##### Transcational Data
The raw market data can be download from https://drive.google.com/file/d/14B3EubjMzNYGAYDkTEIMeTJ3weXQycw6/view?usp=sharing.
Please refer to the notebook "rawdata/marketdata_preprocessing.ipynb" for preprocessing. 

##### Sentiment Indicators
The news data are not provided in our repository due to copyright issues.

We use financial news from Reuters and Bloomberg over the period from 2011 to 2013, released by Ding et al. ("Using structured events to predict stock price movement: An empirical investigation." EMNLP. 2014.)  

The Loughran-McDonald Master Dictionary (https://sraf.nd.edu/textual-analysis/resources/) is used to extract sentiment from financial articles.

##### Company Relations
Firm relations can be found at ./raw_data/relations/*.
The Company Relations are collected from S&P Capical IQ (https://www.capitaliq.com/). 
The first row is the target stock tickers, and following rows are firms that has specific relation with that frim.

## Contact
chengrui0108@hotmail.com
