FROM pure/python:3.6-cuda10.1-cudnn7-runtime

RUN pip install -U torch==1.4.0 torchvision==0.5.0
RUN pip install cython
RUN pip install opencv-python pillow pycocotools matplotlib 
RUN pip install onnxruntime-gpu
RUN pip install onnx