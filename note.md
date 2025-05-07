
# 1.导出PaddleOCR的onnx模型

## 1.1 克隆项目到本地。

> git clone https://github.com/PaddlePaddle/PaddleOCR.git --depth=1

## 1.2 配置运行环境

```sh
> conda create -n paddle python=3.10
> 
> conda activate paddle
> 
> python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
> 
> pip install -r requirements.txt
```

## 1.3 下载paddleocr的预训练模型

所有模型目录：https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_detection.html

### 高精度版本
1.文本检测模型
+ 模型文件 https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar


2.文本方向分类模型
+ 模型文件: https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar


3.文本识别模型(中英,6k+字符) 
+ 模型文件 https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar

4.文本识别模型(中繁日英,15k+字符)
+ 模型文件 https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv4_server_rec_doc_infer.tar

### 轻量化版本
1.文本检测模型
+ 模型文件: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar

2.文本方向分类模型
+ 模型文件: https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar

3.文本识别模型(中英,6k+字符) 
+ 模型文件 https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar


## 1.4 导出为onnx模型使用

+ paddle2onnx将推理模型转化为onnx模型

```sh
> pip install paddle2onnx
```

1.文本检测模型
```sh
paddle2onnx \
    --model_dir ./pretrain/det/server/ \
    --save_file ./output/det/det_server.onnx \
    -mf inference.json \
    -pf inference.pdiparams\
    --opset_version 17\
    --enable_onnx_checker True
    
paddle2onnx \
    --model_dir ./pretrain/det/mobile/ \
    --save_file ./output/det/det_mobile.onnx \
    -mf inference.json \
    -pf inference.pdiparams\
    --opset_version 17\
    --enable_onnx_checker True
    
```

2.文本方向分类模型

```sh
paddle2onnx \
    --model_dir ./pretrain/cls/ \
    --save_file ./output/cls/cls.onnx \
    -mf inference.json \
    -pf inference.pdiparams\
    --opset_version 17\
    --enable_onnx_checker True
    
```

3.文本识别模型


```sh
paddle2onnx \
    --model_dir ./pretrain/rec/mobile/ \
    --save_file ./output/rec/rec_zh_mobile.onnx \
    -mf inference.json \
    -pf inference.pdiparams\
    --opset_version 17\
    --enable_onnx_checker True
    
paddle2onnx \
    --model_dir ./pretrain/rec/server/ \
    --save_file ./output/rec/rec_zh_server.onnx \
    -mf inference.json \
    -pf inference.pdiparams\
    --opset_version 17\
    --enable_onnx_checker True

paddle2onnx \
    --model_dir ./pretrain/rec/server_doc/ \
    --save_file ./output/rec/rec_zh_server_doc.onnx \
    -mf inference.json \
    -pf inference.pdiparams\
    --opset_version 17\
    --enable_onnx_checker True
    
```











