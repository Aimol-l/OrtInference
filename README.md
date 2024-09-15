# OrtInference
使用 C++和onnxruntime框架进行多种任务的推理。

目前完成：
 + YoloV10 目标检测
 + YoloV10 + SAM 检测出目标并进行分割出mask
 + YoloV10 + ByteTrack 视频中多目标跟踪
 + SAM2 视频的目标跟踪分割

SAM2 所用的onnx文件，参考https://github.com/Aimol-l/SAM2Export

## 依赖
+ opencv >= 4.8,tested on 4.10
+ onnxruntime >= 1.18.0,tested on 1.18.1
+ cuda >= 12.x,tested on 12.5
+ cudnn >= 9.x,tested on 9.2
+ gcc >= 14.1.1 (required c++23)
+ cmake >= 3.25,tested on 3.30
+ tbb, tested on 2021.13.0

## Arch Linux

```sh
sudo pacman -S opencv cmake onetbb cuda cudnn onnxruntime 
```
## 使用

```sh
cmake -B build
cd build
cmake .. && make && ../bin/./main
```

## video

[test.mp4](https://www.acfun.cn/v/ac45502468)

[Yolov10 C++ Opencv OnnxRuntime推理部署](https://www.acfun.cn/v/ac45473033?shareUid=31449214)

[Yolov10+SAM C++ Opencv OnnxRuntime推理部署](https://www.acfun.cn/v/ac45487564?shareUid=31449214)

[Yolov10+ByteTrack C++ Opencv OnnxRuntime推理部署](https://www.acfun.cn/v/ac45658815)

[SAM2 C++ Opencv OnnxRuntime推理部署](https://www.acfun.cn/v/ac46243626)
## 参考

SAM：https://github.com/facebookresearch/segment-anything

EfficientSAM： https://github.com/yformer/EfficientSAM

YOLOv10：https://github.com/THU-MIG/yolov10

SAM2Export：https://github.com/Aimol-l/SAM2Export

SAM2：https://github.com/facebookresearch/segment-anything-2
