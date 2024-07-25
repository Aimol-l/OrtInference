#include <iostream>
#include <filesystem>
#include "Yolov10.h"
#include "Yolov10SAM.h"

int main(int argc, char const *argv[]){
    auto yolov10sam = std::make_unique<Yolov10SAM>();
    
    std::string onnx_path = "../models/yolov10m.onnx|../models/ESAM_encoder.onnx|../models/ESAM_deocder.onnx";
    yolov10sam->initialize(onnx_path,true);
    yolov10sam->setparms({.score=0.5f,.nms=0.8f});
    
    std::string folder_path = "../assets/input/*.jpg";
    std::string output_path = "../assets/output/";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto& path : paths) {
        std::println("path={}",path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        int r = yolov10sam->inference(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("duration = {}ms",duration);
        if(r){
            auto filename = std::filesystem::path(path).filename().string();
            cv::imwrite(output_path+filename,image);
            cv::imshow("Image", image);
            cv::waitKey(0);
        }else{
            std::println("inference error!!!");
            continue;
        }
    }
    return 0;   
}

