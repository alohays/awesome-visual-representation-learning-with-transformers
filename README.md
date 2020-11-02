# Awesome Visual Representation Learning with Transformers

## About transformers
- Attention Is All You Need, NeurIPS 2017
  - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
  - [[paper]](https://arxiv.org/abs/1706.03762) [[official code]](https://github.com/tensorflow/tensor2tensor) [[pytorch implementation]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019
  - Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
  - [[paper]](https://arxiv.org/abs/1810.04805) [[offficial code]](https://github.com/google-research/bert) [[huggingface/transformers]](https://github.com/huggingface/transformers)
  
## Combining CNN with self-attention
- Attention augmented convolutional networks, ICCV 2019, image classification
  - Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, Quoc V. Le
  - [[paper]](https://arxiv.org/abs/1904.09925) [[pytorch implementation]](https://github.com/leaderj1001/Attention-Augmented-Conv2d)
- End-to-end object detection with transformers, ECCV 2020, object detection
  - Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
  - [[paper]](https://arxiv.org/abs/2005.12872) [[official code]](https://github.com/facebookresearch/detr)
- Videobert: A joint model for video and language representation learning, ICCV 2019, video processing
  - Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, Cordelia Schmid
  - [[paper]](https://arxiv.org/abs/1904.01766)
- Visual Transformers: Token-based Image Representation and Processing for Computer Vision, arxiv 2020, image classification
  - Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, Peter Vajda
  - [[paper]](https://arxiv.org/abs/2006.03677)

## Stand-alone transformers for Computer Vision
### Self-attention only in local neighborhood
- Image Transformer, ICML 2018
  - Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Łukasz Kaiser, Noam Shazeer, Alexander Ku, Dustin Tran
  - [[paper]](https://arxiv.org/abs/1802.05751) [[official code]](https://github.com/tensorflow/tensor2tensor)
- Stand-alone self-attention in vision models, NeurIPS 2019
  - Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, Jonathon Shlens
  - [[paper]](https://arxiv.org/abs/1906.05909) [[official code(underconstruction)]](https://github.com/google-research/google-research/tree/master/standalone_self_attention_in_vision_models)
- On the relationship between self-attention and convolutional layers, ICLR 2020
  - Jean-Baptiste Cordonnier, Andreas Loukas, Martin Jaggi
  - [[paper]](https://arxiv.org/abs/1911.03584) [[official code]](https://github.com/epfml/attention-cnn)
- Exploring self-attention for image recognition, CVPR 2020
  - Hengshuang Zhao, Jiaya Jia, Vladlen Koltun
  - [[paper]](https://arxiv.org/abs/2004.13621) [[official code]](https://github.com/hszhao/SAN)
### Scalable approximations to global self-attention
- Generating long sequences with sparse transformers, arXiv 2019
- Scaling autoregressive video models, ICLR 2019
- Axial attention in multidimensional transformers, arXiv 2019
- Axial-deeplab: Stand-alone axial-attention for panoptic segmentation, ECCV 2020
### Global self-attention with image preprocessing
- Generative pretraining from pixels, ICML 2020, iGPT
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, arXiv 2020, ViT

## Unified text-vision tasks
### Focused on VQA
### Focused on Image Retrieval
### Focused on OCR
