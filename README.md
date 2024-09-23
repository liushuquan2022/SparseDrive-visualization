# SparseDrive-v: 基于视觉稀疏场景的端到端自动驾驶


## 介绍
> SparseDrive 由对称稀疏感知模块和并行运动规划器组成，是一种以 Sparse 为中心的端到端自动驾驶范例。SparseDrive利用解耦的实例特征和几何锚点作为一个实例（动态道路智能体或静态地图元素）的完整表示，对称稀疏感知将检测、跟踪和在线地图任务与对称模型架构统一起来，实现完全稀疏的场景表示。在并行运动规划器中，首先从自我实例初始化模块获得语义和几何感知的自我实例。利用来自稀疏感知的自我实例和周围智能体实例，同时进行运动预测和规划，以获得所有道路智能体的多模态轨迹。为了确保规划的合理性和安全性，采用了结合碰撞感知重新评分模块的分层规划选择策略，从多模态轨迹提案中选择最终的安全规划轨迹。详情请参见作者的论文：[arXiv](https://arxiv.org/abs/2405.19620). 本文的主要贡献：
- 在 SparseDrive 项目基础上实现了检测、运动规划、预测轨迹、可视化输出的一体化集成.
- 单GPU、小内存性能优化.
- 多视角观察可视化输出，图形文字的中文显示.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/overview.png" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>
 
[视频](demo/bev_gt.mp4)

### - 直行： 
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="demo/0028.jpg" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

### - 左转： 
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="demo/0039.jpg" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

### - 路口： 
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="demo/0323.jpg" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

### - 右转： 
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="demo/0429.jpg" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

### - 左转： 
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="demo/0636.jpg" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center>

## 性能：
| Model | config | ckpt | log | det: NDS | mapping: mAP | track: AMOTA |track: AMOTP | motion: EPA_car |motion: minADE_car| motion: minFDE_car | motion: MissRate_car | planning: CR | planning: L2 |
| :---: | :---: | :---: | :---: | :---: | :---:|:---:|:---: | :---: | :----: | :----: | :----: | :----: | :----: |
| Stage1 |[cfg](projects/configs/sparsedrive_small_stage1.py)|[ckpt](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth)|[log](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1_log.txt)|0.5260|0.5689|0.385|1.260| | | | | | |
| Stage2 |[cfg](projects/configs/sparsedrive_small_stage2.py)|[ckpt](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth)|[log](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2_log.txt)|0.5257|0.5656|0.372|1.248|0.492|0.61|0.95|0.133|0.097%|0.61|

## 规划：
| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.45 | 0.70 | 1.04 | 0.73 | 0.66 | 0.66 | 0.72 | 0.68 |
| UniAD-wo-post-optim | 0.32 | 0.58 | 0.94 | 0.61 | 0.17 | 0.27 | 0.42 | 0.29 |
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.03 | 0.21 | 0.49 | 0.24 | 
| SparseDrive-S | 0.30 | 0.58 | 0.95 | 0.61 | 0.01 | 0.05 | 0.23 | 0.10 | 


## 开始
[环境安装及数据准备](docs/quick_start.md)

## 引用
如果您发现 SparseDrive-v 对您的研究或应用有用，在下面的 BibTeX 条目中引用它。
```
@article{liu518 2024sparsedrive,
  title={SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation},
  author={liushuquan},
  year={2024}
}
```

## 本文参考如下项目，在此感谢
- [SparseDrive](https://github.com/swc-17/SparseDrive)
- [VAD](https://github.com/hustvl/VAD)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

