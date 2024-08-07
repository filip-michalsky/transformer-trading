# Blockhouse Assignment Report

Author: Filip Michalsky

Preambule: Hi! This was a lot of fun. Thanks for giving me this task. My repo is [here](https://github.com/filip-michalsky/transformer-trading).

## Executive Summary

In this task, I have implemented a Transformer-based architecture utilizing the PPO training approach. I utilized [Google Collab](https://colab.research.google.com/drive/1J0KY0h-f6akrG-VAO-sl5d_ORIaNN8gl?usp=sharing) and used L4 GPU with 23GB of RAM. I have compared my results with a vanilla Transformer approach, Blockhouse-provided mean reversion bot (blotter) and a PPO model from stable baselines, as well as a simple momentum strategy. My model beat the PPO model benchmark provided by Blockhouse and the blotter mean reversion simple trading strategy, but underperformed a simple momentum strategy (implemented by me).

## Evaluation

I have evaluated all of the algorithms on the held out test set (last 30% of the time sequence for the day).

- Vanilla Transfomer: Did not learn to trade
- Transformer + PPO - my implementation: 
    - Cumulative reward: -3174
    - Portfolio Value at Market Close: $10,006,779.095
- PPO Agent from Stable Baselines:
    - Cumulative reward: -2651
    - Portfolio Value at Market Close: $10,000,321.125
- Simple Mean Reversion (blotter):
    - Cumulative reward: -3392
    - Portfolio Value at Market Close: $9,941,926.795
- Simple Momentum - my implementation:
    - Portfolio Value at Market Close: $10,049,549.64
    

## Improvements Potential

I only had ~3 days to implement this while also working full-time.
Example directions I would iterate on:
- Feature Engineering: Include additional data from the trading history and create lagging indicator features.
- Do more hyperparameter and architecture search.
- Increase the dataset size.

Also some cool improvement directions from Claude:
1. Attention Visualization: Implement attention visualization to understand what the model is focusing on when making decisions.
2. Multi-step Returns: Use multi-step returns instead of single-step returns for more stable learning.
3. Curiosity-driven Exploration: Implement intrinsic rewards based on prediction error to encourage exploration.
4. Prioritized Experience Replay: Implement prioritized experience replay to focus on important transitions.
5. Ensemble Methods: Use an ensemble of models to make more robust predictions.
6. Curriculum Learning: Start with simpler trading scenarios and gradually increase complexity.
7. Meta-learning: Implement meta-learning techniques to adapt quickly to market changes.
8. Risk-aware Objectives: Incorporate risk measures (e.g., Sharpe ratio) directly into the objective function.
9. Hierarchical RL: Implement a hierarchical structure with high-level strategy and low-level execution agents.
10. Multi-agent Learning: Extend to multi-agent scenarios to model complex market dynamics.
11. Adversarial Training: Use adversarial examples to make the model more robust to market manipulations.
Interpretability: Implement techniques like SHAP values to explain model decisions.

## Technical Approach

I started with a review of the task and the dataset and did some [EDA](https://chatgpt.com/share/aaf775a4-d960-49e3-b835-04af78a8f9bb).

The market trades dataset was split into train and test to prevent information leakage. All algorithms benchmarked in this work were only trained on training set (from market open till roughly 3PM) and evaluated on trading data from 3PM till close (test set).

I then started a vanilla transformer implementation, tweaked it to better work with numerical continuous values and did a lot of debugging to prevent exploding gradients.

I then move on to combining the transformer architecture with PPO reinforcement learning strategy - where the actor agent proposes trade recommendations and critic predicts value of them and then they each have a separate loss function (this is not dissimilar from training GANs and looking for saddle points).

Challenges I had to overcome:

- Numerical instability: I had to dig deep to set up the right architecture which would not explode gradients on me. I utilized batch layer normalization, gradient clipping, early stopping, learning rate adjustments and robust monitoring to overcome this issue.
- Train/test split - this is basic, but the originally assignment was overfitting the PPO model since it included the first 10k training steps in back-testing.
- Model "underfitting": transformers are 'data-hungry' models and we are feeding in low-dimensional sequential data.

My final architecture attempted to reduce overfitting by using a more lightweight footprint with less attention heads, small number of layers, encoding actor embedding to 32 dimensional latent vector, not running through the batches multiple times, using bigger batch size (128) for gradient accumulation and using drop out.

Final Model Architecture:

(note that we only use the actor for inference)

```plaintext
TransformerPPOActor(
   (embedding): Sequential(
     (0): Linear(in_features=17, out_features=32, bias=True)
     (1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
   )
   (transformer_encoder): TransformerEncoder(
     (layers): ModuleList(
       (0-1): 2 x TransformerEncoderLayer(
         (self_attn): MultiheadAttention(
           (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
         )
         (linear1): Linear(in_features=32, out_features=2048, bias=True)
         (dropout): Dropout(p=0.3, inplace=False)
         (linear2): Linear(in_features=2048, out_features=32, bias=True)
         (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
         (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
         (dropout1): Dropout(p=0.3, inplace=False)
         (dropout2): Dropout(p=0.3, inplace=False)
       )
     )
   )
   (fc): Linear(in_features=32, out_features=3, bias=True)
 ),
 TransformerPPOCritic(
   (embedding): Sequential(
     (0): Linear(in_features=17, out_features=64, bias=True)
     (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
   )
   (transformer_encoder): TransformerEncoder(
     (layers): ModuleList(
       (0-1): 2 x TransformerEncoderLayer(
         (self_attn): MultiheadAttention(
           (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
         )
         (linear1): Linear(in_features=64, out_features=2048, bias=True)
         (dropout): Dropout(p=0.3, inplace=False)
         (linear2): Linear(in_features=2048, out_features=64, bias=True)
         (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
         (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
         (dropout1): Dropout(p=0.3, inplace=False)
         (dropout2): Dropout(p=0.3, inplace=False)
       )
     )
   )
   (fc): Linear(in_features=64, out_features=1, bias=True)
 )
```

Example recommendations with our PPO+Transformer setup (index is of the test set which is the last 30% of the trade events):

```plaintext
=== Hold Recommendations ===
Trade Recommendation for AAPL at index 2362:
--------------------------------------------------
Input Data:
       Close Volume  RSI      MACD MACD_signal MACD_hist Stoch_k Stoch_d  \
2362  192.28    100  0.0  0.003045    0.005822 -0.002777     0.0     0.0   

           OBV    Upper_BB  Middle_BB    Lower_BB ATR_1       ADX       +DI  \
2362  912052.0  192.298419  192.28725  192.276081  0.01  50.40793  6.955222   

           -DI  CCI  
2362  4.606997  0.0  

Recommendation Probabilities:
Hold: 36.16%, Buy: 28.72%, Sell: 35.12%

Recommended Action: Hold
==================================================
Trade Recommendation for AAPL at index 5195:
--------------------------------------------------
Input Data:
       Close Volume    RSI      MACD MACD_signal MACD_hist    Stoch_k  \
5195  192.23    100  100.0  0.004312     0.00405  0.000262  88.888889   

        Stoch_d       OBV    Upper_BB Middle_BB    Lower_BB ATR_1        ADX  \
5195  96.296296  915926.0  192.236577  192.2235  192.210423  0.02  30.850894   

            +DI       -DI         CCI  
5195  12.699157  2.605216  166.666667  

Recommendation Probabilities:
Hold: 38.43%, Buy: 32.87%, Sell: 28.70%

Recommended Action: Hold
==================================================
Trade Recommendation for AAPL at index 12386:
--------------------------------------------------
Input Data:
        Close Volume  RSI      MACD MACD_signal MACD_hist    Stoch_k  \
12386  192.34    101  0.0 -0.006959   -0.006609 -0.000349  22.222222   

         Stoch_d       OBV    Upper_BB  Middle_BB    Lower_BB ATR_1  \
12386  11.111111  915976.0  192.363342  192.34975  192.336158  0.01   

             ADX       +DI        -DI        CCI  
12386  82.745639  0.151837  14.850535  41.666667  

Recommendation Probabilities:
Hold: 36.85%, Buy: 32.53%, Sell: 30.63%

Recommended Action: Hold
==================================================

=== Buy Recommendations ===
Trade Recommendation for AAPL at index 15453:
--------------------------------------------------
Input Data:
        Close Volume    RSI      MACD MACD_signal MACD_hist Stoch_k  \
15453  192.53     56  100.0  0.003865    0.003049  0.000816   100.0   

         Stoch_d       OBV   Upper_BB Middle_BB   Lower_BB ATR_1        ADX  \
15453  97.222222  911903.0  192.53104  192.5215  192.51196  0.01  99.212838   

            +DI       -DI        CCI  
15453  6.194779  0.000077  41.666667  

Recommendation Probabilities:
Hold: 25.97%, Buy: 38.34%, Sell: 35.69%

Recommended Action: Buy
==================================================
Trade Recommendation for AAPL at index 305:
--------------------------------------------------
Input Data:
     Close Volume   RSI      MACD MACD_signal MACD_hist Stoch_k Stoch_d  \
305  192.3     18  50.0  0.005176    0.006642 -0.001466    50.0    50.0   

          OBV    Upper_BB  Middle_BB    Lower_BB ATR_1        ADX        +DI  \
305  912570.0  192.301931  192.29975  192.297569  0.01  57.146815  17.782686   

          -DI        CCI  
305  2.246297 -55.555556  

Recommendation Probabilities:
Hold: 25.19%, Buy: 38.78%, Sell: 36.04%

Recommended Action: Buy
==================================================
Trade Recommendation for AAPL at index 5074:
--------------------------------------------------
Input Data:
       Close Volume  RSI      MACD MACD_signal MACD_hist Stoch_k Stoch_d  \
5074  192.21    300  0.0  0.000136    0.001949 -0.001813     0.0     0.0   

           OBV    Upper_BB  Middle_BB    Lower_BB ATR_1        ADX       +DI  \
5074  915831.0  192.225896  192.21725  192.208604  0.01  22.732038  5.823186   

           -DI  CCI  
5074  5.146688  0.0  

Recommendation Probabilities:
Hold: 24.18%, Buy: 41.15%, Sell: 34.67%

Recommended Action: Buy
==================================================

=== Sell Recommendations ===
Trade Recommendation for AAPL at index 16060:
--------------------------------------------------
Input Data:
       Close Volume  RSI      MACD MACD_signal MACD_hist Stoch_k Stoch_d  \
16060  192.5    100  NaN -0.002673   -0.003109  0.000435     0.0     0.0   

            OBV    Upper_BB Middle_BB    Lower_BB ATR_1        ADX       +DI  \
16060  911779.0  192.508642  192.5015  192.494358  0.01  59.855898  0.430029   

            -DI  CCI  
16060  1.719474  0.0  

Recommendation Probabilities:
Hold: 28.08%, Buy: 35.56%, Sell: 36.35%

Recommended Action: Sell
==================================================
Trade Recommendation for AAPL at index 2191:
--------------------------------------------------
Input Data:
       Close Volume  RSI     MACD MACD_signal MACD_hist Stoch_k Stoch_d  \
2191  192.25     94  NaN  0.00946    0.011751 -0.002291    50.0    50.0   

           OBV    Upper_BB Middle_BB    Lower_BB ATR_1        ADX       +DI  \
2191  911508.0  192.253859  192.2495  192.245141  0.01  50.652451  11.37401   

          -DI        CCI  
2191  7.00763  55.555556  

Recommendation Probabilities:
Hold: 25.38%, Buy: 37.18%, Sell: 37.44%

Recommended Action: Sell
==================================================
Trade Recommendation for AAPL at index 6564:
--------------------------------------------------
Input Data:
      Close Volume  RSI      MACD MACD_signal MACD_hist Stoch_k Stoch_d  \
6564  192.1     50  NaN -0.003319   -0.004136  0.000816     0.0     0.0   

           OBV    Upper_BB Middle_BB    Lower_BB ATR_1        ADX       +DI  \
6564  917744.0  192.103501  192.1005  192.097499  0.01  73.259896  0.338017   

          -DI  CCI  
6564  3.03933  0.0  

Recommendation Probabilities:
Hold: 30.06%, Buy: 33.83%, Sell: 36.11%

Recommended Action: Sell
==================================================
```

Happy to discuss the technical details of my approach further, just let me know.