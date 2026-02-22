# Protego: User-Centric Pose-Invariant Privacy Protection Against Face Recognition-Induced Digital Footprint Exposure

![](assets/intro-git.png)
**Abstract**: Face recognition (FR) technologies are increasingly used to power large-scale image retrieval systems, raising serious privacy concerns. Services like Clearview AI and PimEyes allow anyone to upload a facial photo and retrieve a large amount of online content associated with that person. This not only enables identity inference but also exposes their digital footprint, such as social media activity, private photos, and news reports, often without their consent. In response to this emerging threat, we propose Protego, a user-centric privacy protection method that safeguards facial images from such retrieval-based privacy intrusions. Protego encapsulates a user’s 3D facial signatures into a pose-invariant 2D representation, which is dynamically deformed into a natural-looking 3D mask tailored to the pose and expression of any facial image of the user, and applied prior to online sharing. Motivated by a critical limitation of existing methods, Protego amplifies the sensitivity of FR models so that protected images cannot be matched even among themselves. Experiments show that Protego significantly reduces retrieval accuracy across a wide range of black-box FR models and performs at least 2x better than existing methods. It also offers unprecedented visual coherence, particularly in video settings where consistency and natural appearance are essential. Overall, Protego contributes to the fight against the misuse of FR for mass surveillance and unsolicited identity tracing.

**Example**: We extract a frame from an interview video of Bradley Cooper and submit it to two platforms: (i) PimEyes, a well-known face search engine, and (ii) Google Images. The search is performed both with and without applying Protego’s protection.
* [Left] Without protection, both platforms successfully identify Bradley Cooper and even retrieve the exact interview video available online.
* [Right] With Protego applied, neither PimEyes nor Google Images is able to find any matches.

The original video and its protected versions using three different methods are shown below. Please note that the GIFs may take a moment to load.

![](assets/banner.png)

| Method              | Original Video \| Protected Video \| Protection Mask |
|---------------------|------------------------------------------------------|
| Protego (Ours)      | ![](assets/demo-bc-protego.gif)                      |
| Chameleon [ECCV'24] | ![](assets/demo-bc-chameleon.gif)                    |
| OPOM [TPAMI'22]     | ![](assets/demo-bc-opom.gif)                         |

For more technical details and experimental results, we invite you to check out our paper [[here]](https://arxiv.org/abs/2508.02034):  
**Ziling Wang, Shuya Yang, Jialin Lu, and Ka-Ho Chow,** *"Protego: User-Centric Pose-Invariant Privacy Protection Against Face Recognition-Induced Digital Footprint Exposure,"*  IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Denver, CO, USA, Jun. 3-7, 2026.
```
@inproceedings{wang2026protego,
    title={Protego: User-Centric Pose-Invariant Privacy Protection Against Face Recognition-Induced Digital Footprint Exposure},
    author={Ziling Wang and Shuya Yang and Jialin Lu and Ka-Ho Chow},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2026}
}
```
---
We provide the code for protecting images and videos with the privacy protection textures (PPTs) and the code for evaluating the protection performance. This repository comes with some pretrained PPTs for demonstration. The training code to generate your own PPT will be released later.

The current version of Protego is not optimized for performance, so there may be OOM issues when running on low-memory devices. For reference, to run Protego, you need around 8 GB of (GPU) memory. The code is tested on: 
- Ubuntu 22.04; Intel Xeon w5-3415 CPU; 1 NVIDIA RTX 5880 Ada GPU (48 GB memory); 128 GB RAM
- MacOS 15.6.1; Apple M4 Pro; 24 GB Memory

Although we do support MPS, the performance and stability are poorer, which is mostly because of PyTorch and PyTorch3D's immature support of MPS. Therefore, we recommend either CPU-only or CUDA-enabled environments for optimal stability or performance. 

For Windows users, you may need to figure out the setup process yourself, especially the PyTorch3D installation part. We will add support for Windows afterwards if more people are interested in the project and would like to try out for themselves. 

## Quick Start from Preprocessed Datasets
0. Clone this repository with
```commandline
$ git clone --depth 1 https://github.com/HKU-TASR/Protego.git
```
1. Run the following commands in the base environment of conda to quickly set up the environment and download the most essential assets:
```commandline
$ bash setup_quick.sh
$ conda activate protego
```
Note that `setup_quick.sh` will automatically run differently according to platform.

2. Launch the notebook `protego.ipynb` in the conda environment and try out the PPTs.
- If you are using VS Code, you can directly choose the kernel to use. 
- Otherwise, try the following command and open `protego.ipynb` (untested)
```commandline
$ conda activate protego
$ conda install jupyter -y
$ jupyter notebook
```

## Acknowledgements
The codes and weights in the following folders are adapted from existing open-source projects:
- [smirk/](https://github.com/georgeretsi/smirk)
- [mtcnn_pytorch/](https://github.com/Michael-wzl/mtcnn_pytorch)
- [DiffJPEG/](https://github.com/mlomnitz/DiffJPEG)
- [FR_DB/adaface/](https://github.com/mk-minchul/AdaFace)
- [FR_DB/arcface/](https://github.com/ronghuaiyang/arcface-pytorch)
- [FR_DB/facenet/](https://github.com/timesler/facenet-pytorch)
- [FR_DB/ir50_opom/](https://github.com/zhongyy/OPOM)
- [FR_DB/magface/](https://github.com/IrvingMeng/MagFace)
