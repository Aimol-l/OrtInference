# OrtInference
使用 C++和onnxruntime框架进行yolov10的推理 

## 依赖
+ opencv >= 4.8,tested on 4.10
+ onnxruntime >= 1.18.0,tested on 1.18.1
+ cuda >= 12.x,tested on 12.5
+ cudnn >= 9.x,tested on 9.2
+ gcc >= 14.1.1 (required c++23)
+ cmake >= 3.25,tested on 3.30
+ tbb, tested on 2021.13.0
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