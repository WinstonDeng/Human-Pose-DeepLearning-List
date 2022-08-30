# Human-Pose-DeepLearning-List
Some recent (2016-now) Human-Pose related deep learning studies. If you find any errors or problems, please feel free to comment and PR.
Early version is shared on [[Zhihu]](https://zhuanlan.zhihu.com/p/345201439) and [[Mind-map]](https://note.youdao.com/ynoteshare/index.html?id=0f2ed184fb4439b36c61f91938989d8c&type=note&_time=1661513476742).

# Content
- [Human-Pose Estimation](#A)
- [Human-Pose Recognition](#B)
- [Human-Pose Generation](#C)
- [Human-Pose Reconstruction and Rendering (Mesh or Appearance)](#D)
- [Open-source Toolbox](#E)


## <div id="A">Human-Pose Estimation<div>

### 1. monocular single image
#### 2D human pose (heatmap-based)
[Top-down]

Popular papers
  - Simple Pose
  - Mask R-CNN
  - CPM
  - CPN
  - RSN
  - AlphaPose
  - MSPN
  - HRNet
  
Others
  - Graph-PCNN: Two Stage Human Pose Estimation with Graph Pose Refinement
  - Efficient Human Pose Estimation by Learning Deeply Aggregated Representations
  - Fast Human Pose Estimation
  - Learning to Refine Human Pose Estimation
  
[Bottom-up]

PAF Family
  - PifPaf
  - OpenPose
  - Improved PifPaf

Associate Family
  - Associative Embedding
  - HigherHRNet
  - Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation

Graph Family
  - Differentiable Hierarchical Graph Grouping for Multi-Person Pose Estimation

Offset Family
  - Personlab

[One-stage]
- Deeply Learned Compositional Models for Human Pose Estimation
- Single-Stage Multi-Person Pose Machines
  
[Interaction-aware]
- I^2RNet

#### 2D human pose (regression-based)
  - Integral Human Pose Regression
  - LCR-Net: Localization-Classification-Regression for Human Pose
  
#### 2D human pose (vector-based)
  
#### 3D human pose
[Non-rigid Structure from Motion]
  - c3dpo
  - Deep Interpretable Non-Rigid Structure from Motion
  
[2D lift to 3D]
  - A simple yet effective baseline for 3d human pose estimation
  
[Depth-aware]
  - HMOR
  
[Others]
  - Unsupervised Cross-Modal Alignment for Multi-Person 3D Pose Estimation
  - PI-Net:  Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation
  - Semantic Graph Convolutional Networks for 3D Human Pose Regression

### 2. heatmap-aware
- UDP
- DarkPose
- Rethink
  
### 3. occlusion-aware
#### GAN
  - Adversarial PoseNet
  - Adversarial Semantic Data Augmentation for Human Pose Estimation
  
#### Post-processing
  - CrowdPose
  - Peeking into occluded joints
  
#### Data augmentation
  - Adversarial Semantic Data Augmentation for Human Pose Estimation
  - 3D Human Pose Estimation using Spatio-Temporal Networks with Explicit Occlusion Training
  - Occlusion-Aware Networks for 3D Human Pose Estimation in Video
  - Occlusion-Aware Siamese Network for Human Pose Estimation
  
#### Others
  - A Semantic Occlusion Model for Human Pose Estimation from a Single Depth
  - Occluded Joints Recovery in 3D Human Pose Estimation based on Distance Matrix
  - Object-Occluded Human Shape and Pose Estimation from a Single Color Image
  - LCR-Net Localization-Classification-Regression for Human Pose

### 4. multi-view images
#### Cross-view
  - Adaptive Multiview Fusion for Accurate Human Pose Estimation in the wild
  - Cross View Fusion for 3D Human Pose Estimation
  - Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS
  - Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views
  - Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation A Geometric Approach
  - View-Invariant Probabilistic Embedding for Human Pose
  
#### Mirror

### 5. video
#### 2D
  - Combining detection and tracking for human pose estimation in videos
  - Key Frame Proposal Network for Efficient Pose Estimation in Videos
  - Learning Temporal Pose Estimation from Sparsely-Labeled Videos
  - Personalizing Human Video Pose Estimation
  - Self-supervised Keypoint Correspondences for Multi-Person Pose Estimation and Tracking in Videos
  
#### 3D
  - VideoPose3D
  - PoseNet3D
  - Motion Guided 3D Pose Estimation from Videos
  - 2D or 3D Pose Estimation and Action Recognition using Multitask Deep Learning
  - 3D Human Pose Estimation from Monocular Video
  - Attention Mechanism Exploits Temporal Contexts Real-time 3D Human Pose Reconstruction
  - VNect

### 6. RGBD image
  - 3D Human Pose Estimation in RGBD Images for Robotic Task Learning
  - Human Pose Estimation for RGBD Imagery with Multi-Channel Mixture of Parts and Kinematic Constraints
  
## <div id="B"> Human-Pose Recognition <div>

### 1. action classification
  - 2D or 3D Pose Estimation and Action Recognition using Multitask Deep Learning
  - SlowFast
  - Long-term Feature Bank
  - AlphAction
  - 2D or 3D Pose Estimation and Action Recognition using Multitask Deep Learning
  
### 2. human-object interaction
  - [HOI-Learning-List](https://github.com/DirtyHarryLYL/HOI-Learning-List)

## <div id="C"> Human-Pose Generation <div>

### 1. motion transfer
  - Liquid Warping GAN with Attention: A Unified Framework for Human Image Synthesis
  - FOMM
  - MRAA
  - Thin
  
### 2. audio to pose
#### speech to pose
  - Speech2Gesture: Learning Individual Styles of Conversational Gesture
  - Speech2Video: Synthesis with 3D Skeleton Regularization and Expressive Body Poses
  
#### music to pose
  - Dancing to Music
  - ChoreoNet: Towards Music to Dance Synthesis with
  - LISTEN TO DANCE
  - Music2Dance
  - Audio to Body Dynamics
  - Multi-Instrumentalist Net Unsupervised Generation of Music from Body Movements
  - Dance with Melody: An LSTM-autoencoder Approach to
  
### 3. text to pose
  - Text2Action
  - Language2Pose: Natural Language Grounded Pose Forecasting
  
### 4. multi-model to pose
  - TriModel

## <div id="D"> Human-Pose Reconstruction and Rendering (Mesh or Appearance) <div>

### 1. parametric model
  - Human Mesh Recovery from Monocular Images via a Skeleton-disentangled Representation
  - VIBE: Video Inference for Human Body Pose and Shape Estimation
  - METRO
  
### 2. NeRF model
- Neural Body: Implicit Neural Representations With Structured Latent Codes for Novel View Synthesis of Dynamic Humans (CVPR 2021) [[Project]](https://zju3dv.github.io/neuralbody/) [[Code]](https://github.com/zju3dv/neuralbody) [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.pdf)
- Neural actor: neural free-view synthesis of human actors with pose control (SIGGRAPH Asia 2021) [[Project]](https://vcai.mpi-inf.mpg.de/projects/NeuralActor) [[Code]](https://github.com/lingjie0206/Neural_Actor_Main_Code) [[Paper]](https://arxiv.org/abs/2106.02019)
- Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies (ICCV 2021) [[Project]](https://zju3dv.github.io/animatable_nerf) [[Code]](https://github.com/zju3dv/animatable_nerf) [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Noguchi_Neural_Articulated_Radiance_Field_ICCV_2021_paper.pdf)
- A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose (NeurIPS 2021) [[Project]](https://lemonatsu.github.io/anerf/) [[Code]](https://github.com/LemonATsu/A-NeRF) [[Paper]](https://arxiv.org/abs/2102.06199)
- Neural Articulated Radiance Field (ICCV 2021) [[Project]](https://github.com/nogu-atsu/NARF) [[Code]](https://github.com/nogu-atsu/NARF) [[Paper]](https://arxiv.org/abs/2104.03110)
- HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video (CVPR 2022) [[Project]](https://grail.cs.washington.edu/projects/humannerf/) [[Code]](https://github.com/chungyiweng/humannerf) [[Paper]](https://arxiv.org/abs/2201.04127)
- HumanNeRF: Efficiently Generated Human Radiance Field from Sparse Inputs
(CVPR 2022) [[Project]](https://zhaofuq.github.io/humannerf/) [[Code]](https://github.com/zhaofuq/HumanNeRF) [[Paper]](https://github.com/zhaofuq/HumanNeRF)

`From https://github.com/jintaiWang/NeRF-About-Human-Pose-Reconstruction-and-Rendering`

### 3. relighting
- [Human Relighting](https://github.com/QifengDai/Human-Relighting-Learning-List)

# <div id="E"> Open-source Toolbox <div>
- [MMPose](https://github.com/open-mmlab/mmpose)
- [MMAction](https://github.com/open-mmlab/mmaction2)
# Other Related List
Coming soon...
- [ ] [Transformer Human Pose]()
- [ ] [Pose-aware Editing]()
