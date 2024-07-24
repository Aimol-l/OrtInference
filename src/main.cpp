#include <iostream>
#include <filesystem>
#include "Yolov10.h"

int main(int argc, char const *argv[]){
    auto yolov10 = std::make_unique<Yolov10>();

    yolov10->initialize(std::string{"../models/yolov10m.onnx"},true);
    yolov10->setparms({.score=0.5f,.nms=0.5f});
    
    std::string folder_path = "../assets/input/*.jpg";
    std::string output_path = "../assets/output/";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto& path : paths) {
        std::println("path={}",path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        int r = yolov10->inference(image);
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

