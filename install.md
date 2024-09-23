# 安装配置说明

### 创建虚拟环境
```bash
conda create -n SparseDrive-v python=3.8 -y
conda activate sparsedrive-v
```

### 安装
```bash
cd ${sparsedrive-v_path}
pip install --upgrade pip

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install fsspec
pip install mmcv_full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2.0/index.html
pip install mmdet==2.28.2

pip install -r requirement.txt
```

### 编译 CUDA
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### 准备数据集
下载 [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) 1.0 数据集，再下载 CAN bus expansion，完成后创建软链接，帮助你访问本地nuscenes数据.
```bash
cd ${sparsedrive-v_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
or
mklink ./data/nuscenes path/to/nuscenes
```

使用nuscenes_converter.py生成map_annos，默认roi_size为 （30， 60），可以在  tools/data_converter/nuscenes_converter.py文件中修改roi_size.
```bash
cd ${sparsedrive-v_path}

python tools/data_converter/nuscenes_converter.py nuscenes --root-path ./data/nuscenes --canbus ./data/nuscenes --out-dir ./data/infos/ --extra-tag nuscenes --version v1.0

or

python tools/data_converter/nuscenes_converter.py nuscenes --root-path ./data/nuscenes --canbus ./data/nuscenes --out-dir ./data/infos/ --extra-tag nuscenes --version v1.0-mini
```

### 生成几何锚点
```bash
cd ${sparsedrive-v_path}
python tools/kmeans/kmeans_det.py
python tools/kmeans/kmeans_map.py
python tools/kmeans/kmeans_motion.py
python tools/kmeans/kmeans_plan.py
```


### 下载weights
按如下说明下载
```bash
cd ${sparsedrive-v_path}
mkdir ckpt

wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth

wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth

wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth
```

### 训练
```bash
python ./tools/train.py  projects/configs/sparsedrive_small_stage1.py --deterministic
python ./tools/train.py  projects/configs/sparsedrive_small_stage2.py --deterministic
```

### 测试
```bash
# 检测
python ./tools/test.py projects/configs/sparsedrive_small_stage1.py ckpt/sparsedrive_stage1.pth --deterministic --eval bbox
python ./tools/test.py projects/configs/sparsedrive_small_stage1.py ckpt/sparsedrive_stage1.pth --deterministic --eval segm

# 规划
python ./tools/test.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --deterministic --eval bbox
python ./tools/test.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --deterministic --eval segm

#检测+规划+可视化
python ./tools/test.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --deterministic --eval bbox --visual
python ./tools/test.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --deterministic --eval segm --visual

#性能分析
python ./tools/test.py projects/configs/sparsedrive_small_stage1.py ckpt/sparsedrive_stage1.pth --deterministic --eval bbox --analysis
python ./tools/test.py projects/configs/sparsedrive_small_stage1.py ckpt/sparsedrive_stage1.pth --deterministic --eval segm --analysis

python ./tools/test.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --deterministic --eval bbox --analysis
python ./tools/test.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --deterministic --eval segm --analysis

#只输出检测结果
python ./tools/test.py projects/configs/sparsedrive_small_stage1.py ckpt/sparsedrive_stage1.pth --deterministic --eval bbox --result_file ./work_dirs/results.pkl

python ./tools/test.py projects/configs/sparsedrive_small_stage2.py ckpt/sparsedrive_stage2.pth --deterministic --eval bbox --result_file ./work_dirs/results.pkl
```

### 独立可视化检测结果

```
python ./tools/visualization/visualize.py projects/configs/sparsedrive_small_stage1.py --result-path ./work_dirs/results.pkl

python ./tools/visualization/visualize.py projects/configs/sparsedrive_small_stage2.py --result-path ./work_dirs/results.pkl
```
