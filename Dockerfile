FROM  tensorflow/tensorflow:latest-gpu-jupyter

RUN apt install -y libgl1-mesa-glx
RUN python3 -m pip install -U torch==1.4.0 torchvision==0.5.0
RUN python3 -m pip install cython
RUN python3 -m pip install opencv-python pillow pycocotools matplotlib
RUN python3 -m pip install onnxruntime onnx