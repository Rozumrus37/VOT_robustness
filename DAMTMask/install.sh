
echo "=======================Installing basic libraries======================="
sudo apt-get install -y libturbojpeg0
sudo apt-get install -y ninja-build

# install libGL.so.1
sudo apt update
sudo apt install -y libgl1-mesa-glx

# install gcc&g++ 7.4.0
sudo apt-get update
sudo apt-get install -y gcc-7
sudo apt-get install -y g++-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --config gcc

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
sudo update-alternatives --config g++
echo "======================================================================="
echo "=========================Installing python libraries========================="

echo "****************** Installing pytorch ******************"
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch

echo "****************** Installing mmcv ******************"
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html


echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
pip install tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing vot-toolkit python ******************"
pip install git+https://github.com/votchallenge/vot-toolkit-python

echo ""
echo ""
echo "****************** Installing onnx and onnxruntime-gpu ******************"
pip install onnx onnxruntime-gpu==1.6.0

echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm==0.3.2

echo "****************** Installing yacs/einops/thop ******************"
pip install yacs
pip install einops
pip install thop

echo "****************** Installation complete! ******************"
