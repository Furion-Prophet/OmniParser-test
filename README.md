# OmniParser网页布局分析测试

## 简介
OmniParser是一个用于网页布局分析的模型，它能够识别网页中的文本、图像、按钮等元素，并给出每个元素的坐标和类型。该模型基于微软的VisualGLM模型，并使用了多种技术来提高识别的准确性和效率。

## 使用方法
要使用OmniParser，您需要先安装Python和PyTorch。然后，您可以使用以下代码来加载模型并进行分析：

## 安装依赖 
```python
cd OmniParser-test
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

## 下载模型
需要在对应文件夹下下载模型
* 视觉语言大模型，可以做 OCR、目标检测、图像分割和描述
weights/florence2: (https://huggingface.co/microsoft/Florence-2-large)[https://huggingface.co/microsoft/Florence-2-large]
* 对Florence-2-large的模型实例，专门识别图标，对图标语义化
weights/icon_caption_florence: (https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_caption)[https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_caption]
* 基于YOLOv8的图标检测模型
weights/icon_detect: (https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_detect)[https://huggingface.co/microsoft/OmniParser-v2.0/tree/main/icon_detect]

## 部署本地可视化页面
```python
python gradio_demo.py
```
