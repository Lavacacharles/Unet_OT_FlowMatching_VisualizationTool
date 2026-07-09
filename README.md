# A Visual Learning Models Tool for Generative Models based on Flow Matching Objective

Details of the implementation.

Model: UNet

Objective: Optimal Transport Flow Matching - the model learns to turn noise into real images, so intermediate we sample intermediate steps to see how model progressively learns to transform noise into images.

Epochs: 1500

Sampled only 6 steps from: 250, 500, 750, 1000, 1250, 1500

Model size ~ 170 Mb

Layers: Encoder - Sinusoidal Time Encoder - Embeddings Per Class - Spatial Attention Midcoder - Decoder

Sampled from:
- Spatial Attention Midcoder attention maps
- Mid coder enriched features buy the Spatial Attention Midcoder

Sampled images 100 classes per each class

Per each image there are 100 interpolation steps, sampled 6 to see how noise converts real images
