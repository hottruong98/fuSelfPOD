## Prequisites:
conda create --name hot-unipad python=3.8
pip install --upgrade pip==23.3.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index  torch-scatter -f https://data.pyg.org/whl/torch-1.9.1%2Bcu111.html
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 #if FAILED? sudo apt-get install libsparsehash-dev

#mmdet
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html 
pip install mmdet==2.14.0 mmsegmentation==0.14.1 # mmdet3d==0.17.3
pip install tifffile numpy==1.21.0 protobuf==3.20.0 scikit-image pycocotools waymo-open-dataset-tf-2-2-0 nuscenes-devkit==1.0.5 
pip install spconv-cu111 gpustat numba==0.48 scipy pandas matplotlib Cython shapely loguru tqdm future fire yacs jupyterlab 
pip install scikit-image pybind11 tensorboardX tensorboard easydict pyyaml open3d addict pyquaternion awscli timm typing-extensions==4.7.1
python setup.py develop

## Data preparation:
python ./extra_tools/create_data.py nuscenes --root-path /home/user/data/hot/nuscenes_mini/nuscenes --out-dir /home/user/data/hot/nuscenes_mini/nuscenes --extra-tag nuscences

## Training:
pip install yapf==0.40.1 # FormatCode() error
pip install setuptools==59.5.0 # Downgrade from 60.2 -> 59.5, attribute error

bash ./extra_tools/dist_train_ssl.sh

pip install lyft_dataset_sdk networkx plyfile trimesh==2.35.39

