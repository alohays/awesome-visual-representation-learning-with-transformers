# Awesome Visual Representation Learning with Transformers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Awesome Transformers (self-attention) in Computer Vision

## About transformers
- Attention Is All You Need, NeurIPS 2017
  - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
  - [[paper]](https://arxiv.org/abs/1706.03762) [[official code]](https://github.com/tensorflow/tensor2tensor) [[pytorch implementation]](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019
  - Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
  - [[paper]](https://arxiv.org/abs/1810.04805) [[offficial code]](https://github.com/google-research/bert) [[huggingface/transformers]](https://github.com/huggingface/transformers)
- Efficient Transformers: A Survey, arXiv 2020
  - Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler
  - [[paper]](https://arxiv.org/abs/2009.06732)
- A Survey on Visual Transformer, arXiv 2020
  - Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, Zhaohui Yang, Yiman Zhang, Dacheng Tao
  - [[paper]](https://arxiv.org/abs/2012.12556)

## Combining CNN with self-attention
- Attention augmented convolutional networks, ICCV 2019, image classification
  - Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, Quoc V. Le
  - [[paper]](https://arxiv.org/abs/1904.09925) [[pytorch implementation]](https://github.com/leaderj1001/Attention-Augmented-Conv2d)
- Self-Attention Generative Adversarial Networks, ICML 2019, generative model(GANs)
  - Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena
  - [[paper]](https://arxiv.org/abs/1805.08318) [[official code]](https://github.com/heykeetae/Self-Attention-GAN)
- Videobert: A joint model for video and language representation learning, ICCV 2019, video processing
  - Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, Cordelia Schmid
  - [[paper]](https://arxiv.org/abs/1904.01766)
- Visual Transformers: Token-based Image Representation and Processing for Computer Vision, arXiv 2020, image classification
  - Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, Peter Vajda
  - [[paper]](https://arxiv.org/abs/2006.03677)
- Feature Pyramid Transformer, ECCV 2020, detection and segmentation
  - Dong Zhang, Hanwang Zhang, Jinhui Tang, Meng Wang, Xiansheng Hua, Qianru Sun
  - [[paper]](http://arxiv.org/abs/2007.09451) [[official code]](https://github.com/ZHANGDONG-NJUST/FPT)
- Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers, arXiv 2020, depth estimation
  - Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
  - [[paper]](http://arxiv.org/abs/2011.02910) [[official code]](https://github.com/mli0603/stereo-transformer)
- End-to-end Lane Shape Prediction with Transformers, arXiv 2020, lane detection
  - Ruijin Liu, Zejian Yuan, Tie Liu, Zhiliang Xiong
  - [[paper]](http://arxiv.org/abs/2011.04233) [[official code]](https://github.com/liuruijin17/LSTR)
- Taming Transformers for High-Resolution Image Synthesis, arXiv 2020, image synthesis
  - Patrick Esser, Robin Rombach, Bjorn Ommer
  - [[paper]](http://arxiv.org/abs/2012.09841)[[official code]](https://github.com/CompVis/taming-transformers)
- TransPose: Towards Explainable Human Pose Estimation by Transformer, arXiv 2020, pose estimation
  - Sen Yang, Zhibin Quan, Mu Nie, Wankou Yang
  - [[paper]](https://arxiv.org/abs/2012.14214)
- End-to-End Video Instance Segmentation with Transformers, arXiv 2020, video instance segmentation
  - Yuqing Wang, Zhaoliang Xu, Xinlong Wang, Chunhua Shen, Baoshan Cheng, Hao Shen, Huaxia Xia
  - [[paper]](https://arxiv.org/abs/2011.14503)
- TransTrack: Multiple-Object Tracking with Transformer, arXiv 2020, MOT
  - Peize Sun, Yi Jiang, Rufeng Zhang, Enze Xie, Jinkun Cao, Xinting Hu, Tao Kong, Zehuan Yuan, Changhu Wang, Ping Luo
  - [[paper]](https://arxiv.org/abs/2012.15460)[[official code]](https://github.com/PeizeSun/TransTrack)
- TrackFormer: Multi-Object Tracking with Transformers, arXiv 2021, MOT
  - Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, Christoph Feichtenhofer
  - [[paper]](https://arxiv.org/abs/2101.02702)
- Line Segment Detection Using Transformers without Edges, arXiv 2021, line segmentation
  - Yifan Xu, Weijian Xu, David Cheung, Zhuowen Tu
  - [[paper]](https://arxiv.org/abs/2101.01909)
- Segmenting Transparent Object in the Wild with Transformer, arXiv 2021, transparent object segmentation
  - Enze Xie, Wenjia Wang, Wenhai Wang, Peize Sun, Hang Xu, Ding Liang, Ping Luo
  - [[paper]](https://arxiv.org/abs/2101.08461)[[official code]](https://github.com/xieenze/Trans2Seg)
- Bottleneck Transformers for Visual Recognition, arXiv 2021, backbone design
  - Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, Ashish Vaswani
  - [[paper]](http://arxiv.org/abs/2101.11605)

### DETR Family
- End-to-end object detection with transformers, ECCV 2020, object detection
  - Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
  - [[paper]](https://arxiv.org/abs/2005.12872) [[official code]](https://github.com/facebookresearch/detr) [[detectron2 implementation]](https://github.com/poodarchu/DETR.detectron2)
- Deformable DETR: Deformable Transformers for End-to-End Object Detection, arXiv 2020, object detection
  - Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai
  - [[paper]](http://arxiv.org/abs/2010.04159) [[official code]](https://github.com/fundamentalvision/Deformable-DETR)
- End-to-End Object Detection with Adaptive Clustering Transformer, arXiv 2020, object detection
  - Minghang Zheng, Peng Gao, Xiaogang Wang, Hongsheng Li, Hao Dong
  - [[paper]](http://arxiv.org/abs/2011.09315)
- UP-DETR: Unsupervised Pre-training for Object Detection with Transformers, arXiv 2020, object detection
  - Zhigang Dai, Bolun Cai, Yugeng Lin, Junying Chen
  - [[paper]](http://arxiv.org/abs/2011.09094)
- DETR for Pedestrian Detection, arXiv 2020, pedestrian detection
  - Matthieu Lin, Chuming Li, Xingyuan Bu, Ming Sun, Chen Lin, Junjie Yan, Wanli Ouyang, Zhidong Deng
  - [[paper]](http://arxiv.org/abs/2012.06785)

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
  - Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever
  - [[paper]](https://arxiv.org/abs/1904.10509) [[official code]](https://github.com/openai/sparse_attention)
- Scaling autoregressive video models, ICLR 2019
  - Dirk Weissenborn, Oscar Täckström, Jakob Uszkoreit
  - [[paper]](https://arxiv.org/abs/1906.02634) 
- Axial attention in multidimensional transformers, arXiv 2019
  - Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, Tim Salimans
  - [[paper]](https://arxiv.org/abs/1912.12180) [[pytorch implementation]](https://github.com/lucidrains/axial-attention)
- Axial-deeplab: Stand-alone axial-attention for panoptic segmentation, ECCV 2020
  - Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, Liang-Chieh Chen
  - [[paper]](https://arxiv.org/abs/2003.07853) [[pytorch implementation]](https://github.com/csrhddlam/axial-deeplab)
- MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers, arXiv 2020
  - Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen
  - [[paper]](http://arxiv.org/abs/2012.00759)
### Global self-attention with image preprocessing
- Generative pretraining from pixels, ICML 2020, iGPT
  - Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, David Luan, Ilya Sutskever
  - [[paper]](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf) [[official code]](https://github.com/openai/image-gpt)
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, arXiv 2020, ViT**
  - Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
  - [[paper]](https://arxiv.org/abs/2010.11929) [[pytorch implementation]](https://github.com/lucidrains/vit-pytorch)
- Pre-Trained Image Processing Transformer, arXiv, IPT
  - Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, Wen Gao
  - [[paper]](http://arxiv.org/abs/2012.00364)
- Training data-efficient image transformers & distillation through attention, arXiv 2020, DeiT
  - Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Herve Jegou
  - [[paper]](http://arxiv.org/abs/2012.12877)[[official code]](https://github.com/facebookresearch/deit)
- Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers, arXiv 2020, SETR
  - Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip H.S. Torr, Li Zhang
  - [[paper]](http://arxiv.org/abs/2012.15840)[[official code]](https://fudan-zvg.github.io/SETR)
- Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, arXiv 2021, T2T-ViT
  - Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Francis EH Tay, Jiashi Feng, Shuicheng Yan
  - [[paper]](http://arxiv.org/abs/2101.11986)[[official code]](https://github.com/yitu-opensource/T2T-ViT)
### Global self-attention on 3D point clouds
- Point Transformer, arXiv 2020, points classification + part/semantic segmentation
  - Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun
  - [[paper]](http://arxiv.org/abs/2011.00931)

## Unified text-vision tasks
### Focused on VQA
- ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks, NeurIPS 2019
  - Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee
  - [[paper]](https://arxiv.org/abs/1908.02265) [[official code]](https://github.com/facebookresearch/vilbert-multi-task)
- LXMERT: Learning Cross-Modality Encoder Representations from Transformers, EMNLP 2019
  - Hao Tan, Mohit Bansal
  - [[paper]](https://arxiv.org/abs/1908.07490) [[official code]](https://github.com/airsplay/lxmert)
- VisualBERT: A Simple and Performant Baseline for Vision and Language, arXiv 2019
  - Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang
  - [[paper]](https://arxiv.org/abs/1908.03557) [[official code]](https://github.com/uclanlp/visualbert)
- VL-BERT: Pre-training of Generic Visual-Linguistic Representations, ICLR 2020
  - Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, Jifeng Dai
  - [[paper]](https://arxiv.org/abs/1908.08530) [[official code]](https://github.com/jackroos/VL-BERT)
- UNITER: UNiversal Image-TExt Representation Learning, ECCV 2020
  - Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, Jingjing Liu
  - [[paper]](https://arxiv.org/abs/1909.11740) [[official code]](https://github.com/ChenRocks/UNITER)
- ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision
  - Wonjae Kim, Bokyung Son, Ildoo Kim
  - [[paper]](https://arxiv.org/abs/2102.03334)

### Focused on Image Retrieval
- Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training, AAAI 2020
  - Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, Ming Zhou
  - [[paper]](https://arxiv.org/abs/1908.06066) [[official code]](https://github.com/microsoft/Unicoder)
- ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data, arXiv 2020
  - Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, Arun Sacheti
  - [[paper]](https://arxiv.org/abs/2001.07966)
- Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks, ECCV 2020
  - Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao
  - [[paper]](https://arxiv.org/abs/2004.06165) [[official code]](https://github.com/microsoft/Oscar)
### Focused on OCR
- LayoutLM: Pre-training of Text and Layout for Document Image Understanding
  - Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou
  - [[paper]](https://arxiv.org/abs/1912.13318) [[official code]](https://github.com/microsoft/unilm/tree/master/layoutlm)

### Focused on Image Captioning

- CPTR: Full Transformer Network for Image Captioning, arXiv 2021
  - Wei Liu, Sihan Chen, Longteng Guo, Xinxin Zhu, Jing Liu
  - [[paper]](http://arxiv.org/abs/2101.10804)


### Multi-Task
- 12-in-1: Multi-Task Vision and Language Representation Learning
  - Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, Stefan Lee
  - [[paper]](https://arxiv.org/abs/1912.02315) [[official code]](https://github.com/facebookresearch/vilbert-multi-task)
