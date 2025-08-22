# Model Analysis Report

- Total Parameters: 2,471,764
- Estimated Size  : 9.43 MB
- Device          : cpu
- Checkpoint      : best_model.pt
- Avg Gen Time    : 311.38 ms
- Throughput      : 3.21 seq/s
- Validation PPL  : 9.55

## Model Parameter Breakdown

|   # | Layer Name                                         | Param Count   | Size       | DType   |
|-----|----------------------------------------------------|---------------|------------|---------|
|   1 | encoder.emb.tok_emb.weight                         | 804,160       | 3141.25 KB | float32 |
|   2 | encoder.layers.0.attention.w_q.weight              | 256           | 1.00 KB    | float32 |
|   3 | encoder.layers.0.attention.w_q.bias                | 16            | 0.06 KB    | float32 |
|   4 | encoder.layers.0.attention.w_k.weight              | 256           | 1.00 KB    | float32 |
|   5 | encoder.layers.0.attention.w_k.bias                | 16            | 0.06 KB    | float32 |
|   6 | encoder.layers.0.attention.w_v.weight              | 256           | 1.00 KB    | float32 |
|   7 | encoder.layers.0.attention.w_v.bias                | 16            | 0.06 KB    | float32 |
|   8 | encoder.layers.0.attention.w_concat.weight         | 256           | 1.00 KB    | float32 |
|   9 | encoder.layers.0.attention.w_concat.bias           | 16            | 0.06 KB    | float32 |
|  10 | encoder.layers.0.norm1.gamma                       | 16            | 0.06 KB    | float32 |
|  11 | encoder.layers.0.norm1.beta                        | 16            | 0.06 KB    | float32 |
|  12 | encoder.layers.0.ffn.linear1.weight                | 256           | 1.00 KB    | float32 |
|  13 | encoder.layers.0.ffn.linear1.bias                  | 16            | 0.06 KB    | float32 |
|  14 | encoder.layers.0.ffn.linear2.weight                | 256           | 1.00 KB    | float32 |
|  15 | encoder.layers.0.ffn.linear2.bias                  | 16            | 0.06 KB    | float32 |
|  16 | encoder.layers.0.norm2.gamma                       | 16            | 0.06 KB    | float32 |
|  17 | encoder.layers.0.norm2.beta                        | 16            | 0.06 KB    | float32 |
|  18 | encoder.layers.1.attention.w_q.weight              | 256           | 1.00 KB    | float32 |
|  19 | encoder.layers.1.attention.w_q.bias                | 16            | 0.06 KB    | float32 |
|  20 | encoder.layers.1.attention.w_k.weight              | 256           | 1.00 KB    | float32 |
|  21 | encoder.layers.1.attention.w_k.bias                | 16            | 0.06 KB    | float32 |
|  22 | encoder.layers.1.attention.w_v.weight              | 256           | 1.00 KB    | float32 |
|  23 | encoder.layers.1.attention.w_v.bias                | 16            | 0.06 KB    | float32 |
|  24 | encoder.layers.1.attention.w_concat.weight         | 256           | 1.00 KB    | float32 |
|  25 | encoder.layers.1.attention.w_concat.bias           | 16            | 0.06 KB    | float32 |
|  26 | encoder.layers.1.norm1.gamma                       | 16            | 0.06 KB    | float32 |
|  27 | encoder.layers.1.norm1.beta                        | 16            | 0.06 KB    | float32 |
|  28 | encoder.layers.1.ffn.linear1.weight                | 256           | 1.00 KB    | float32 |
|  29 | encoder.layers.1.ffn.linear1.bias                  | 16            | 0.06 KB    | float32 |
|  30 | encoder.layers.1.ffn.linear2.weight                | 256           | 1.00 KB    | float32 |
|  31 | encoder.layers.1.ffn.linear2.bias                  | 16            | 0.06 KB    | float32 |
|  32 | encoder.layers.1.norm2.gamma                       | 16            | 0.06 KB    | float32 |
|  33 | encoder.layers.1.norm2.beta                        | 16            | 0.06 KB    | float32 |
|  34 | decoder.emb.tok_emb.weight                         | 804,160       | 3141.25 KB | float32 |
|  35 | decoder.layers.0.self_attention.w_q.weight         | 256           | 1.00 KB    | float32 |
|  36 | decoder.layers.0.self_attention.w_q.bias           | 16            | 0.06 KB    | float32 |
|  37 | decoder.layers.0.self_attention.w_k.weight         | 256           | 1.00 KB    | float32 |
|  38 | decoder.layers.0.self_attention.w_k.bias           | 16            | 0.06 KB    | float32 |
|  39 | decoder.layers.0.self_attention.w_v.weight         | 256           | 1.00 KB    | float32 |
|  40 | decoder.layers.0.self_attention.w_v.bias           | 16            | 0.06 KB    | float32 |
|  41 | decoder.layers.0.self_attention.w_concat.weight    | 256           | 1.00 KB    | float32 |
|  42 | decoder.layers.0.self_attention.w_concat.bias      | 16            | 0.06 KB    | float32 |
|  43 | decoder.layers.0.norm1.gamma                       | 16            | 0.06 KB    | float32 |
|  44 | decoder.layers.0.norm1.beta                        | 16            | 0.06 KB    | float32 |
|  45 | decoder.layers.0.enc_dec_attention.w_q.weight      | 256           | 1.00 KB    | float32 |
|  46 | decoder.layers.0.enc_dec_attention.w_q.bias        | 16            | 0.06 KB    | float32 |
|  47 | decoder.layers.0.enc_dec_attention.w_k.weight      | 256           | 1.00 KB    | float32 |
|  48 | decoder.layers.0.enc_dec_attention.w_k.bias        | 16            | 0.06 KB    | float32 |
|  49 | decoder.layers.0.enc_dec_attention.w_v.weight      | 256           | 1.00 KB    | float32 |
|  50 | decoder.layers.0.enc_dec_attention.w_v.bias        | 16            | 0.06 KB    | float32 |
|  51 | decoder.layers.0.enc_dec_attention.w_concat.weight | 256           | 1.00 KB    | float32 |
|  52 | decoder.layers.0.enc_dec_attention.w_concat.bias   | 16            | 0.06 KB    | float32 |
|  53 | decoder.layers.0.norm2.gamma                       | 16            | 0.06 KB    | float32 |
|  54 | decoder.layers.0.norm2.beta                        | 16            | 0.06 KB    | float32 |
|  55 | decoder.layers.0.ffn.linear1.weight                | 256           | 1.00 KB    | float32 |
|  56 | decoder.layers.0.ffn.linear1.bias                  | 16            | 0.06 KB    | float32 |
|  57 | decoder.layers.0.ffn.linear2.weight                | 256           | 1.00 KB    | float32 |
|  58 | decoder.layers.0.ffn.linear2.bias                  | 16            | 0.06 KB    | float32 |
|  59 | decoder.layers.0.norm3.gamma                       | 16            | 0.06 KB    | float32 |
|  60 | decoder.layers.0.norm3.beta                        | 16            | 0.06 KB    | float32 |
|  61 | decoder.layers.1.self_attention.w_q.weight         | 256           | 1.00 KB    | float32 |
|  62 | decoder.layers.1.self_attention.w_q.bias           | 16            | 0.06 KB    | float32 |
|  63 | decoder.layers.1.self_attention.w_k.weight         | 256           | 1.00 KB    | float32 |
|  64 | decoder.layers.1.self_attention.w_k.bias           | 16            | 0.06 KB    | float32 |
|  65 | decoder.layers.1.self_attention.w_v.weight         | 256           | 1.00 KB    | float32 |
|  66 | decoder.layers.1.self_attention.w_v.bias           | 16            | 0.06 KB    | float32 |
|  67 | decoder.layers.1.self_attention.w_concat.weight    | 256           | 1.00 KB    | float32 |
|  68 | decoder.layers.1.self_attention.w_concat.bias      | 16            | 0.06 KB    | float32 |
|  69 | decoder.layers.1.norm1.gamma                       | 16            | 0.06 KB    | float32 |
|  70 | decoder.layers.1.norm1.beta                        | 16            | 0.06 KB    | float32 |
|  71 | decoder.layers.1.enc_dec_attention.w_q.weight      | 256           | 1.00 KB    | float32 |
|  72 | decoder.layers.1.enc_dec_attention.w_q.bias        | 16            | 0.06 KB    | float32 |
|  73 | decoder.layers.1.enc_dec_attention.w_k.weight      | 256           | 1.00 KB    | float32 |
|  74 | decoder.layers.1.enc_dec_attention.w_k.bias        | 16            | 0.06 KB    | float32 |
|  75 | decoder.layers.1.enc_dec_attention.w_v.weight      | 256           | 1.00 KB    | float32 |
|  76 | decoder.layers.1.enc_dec_attention.w_v.bias        | 16            | 0.06 KB    | float32 |
|  77 | decoder.layers.1.enc_dec_attention.w_concat.weight | 256           | 1.00 KB    | float32 |
|  78 | decoder.layers.1.enc_dec_attention.w_concat.bias   | 16            | 0.06 KB    | float32 |
|  79 | decoder.layers.1.norm2.gamma                       | 16            | 0.06 KB    | float32 |
|  80 | decoder.layers.1.norm2.beta                        | 16            | 0.06 KB    | float32 |
|  81 | decoder.layers.1.ffn.linear1.weight                | 256           | 1.00 KB    | float32 |
|  82 | decoder.layers.1.ffn.linear1.bias                  | 16            | 0.06 KB    | float32 |
|  83 | decoder.layers.1.ffn.linear2.weight                | 256           | 1.00 KB    | float32 |
|  84 | decoder.layers.1.ffn.linear2.bias                  | 16            | 0.06 KB    | float32 |
|  85 | decoder.layers.1.norm3.gamma                       | 16            | 0.06 KB    | float32 |
|  86 | decoder.layers.1.norm3.beta                        | 16            | 0.06 KB    | float32 |
|  87 | decoder.linear.weight                              | 804,160       | 3141.25 KB | float32 |
|  88 | decoder.linear.bias                                | 50,260        | 196.33 KB  | float32 |

## Graphs

![Runtime](runtime_scaling.png)

![Training](training.png)

