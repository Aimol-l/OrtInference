#include <iostream>
#include <filesystem>
#include "Yolov10.h"
#include "Yolov10SAM.h"
#include "Yolov10Trace.h"
#include "GroundDino.h"

void yolo(){
    auto yolov10 = std::make_unique<Yolov10>();
    
    std::string onnx_path = "../models/yolov10m.onnx";
    yolov10->initialize(onnx_path,true);
    yolov10->setparms({.score=0.5f,.nms=0.8f});
    
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
}
void yolosam(){
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
}
void yolotrace(){
    auto yolov10trace = std::make_unique<Yolov10Trace>();
    
    std::string onnx_path = "../models/yolov10m.onnx";
    yolov10trace->initialize(onnx_path,true);

    std::string video_path = "../assets/video/test.mp4";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) return;
    //************************************************************
    std::cout << "视频中图像的宽度=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << capture.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
    //************************************************************
    yolov10trace->setparms({
        .camera_fps = 25,
        .buffer_size = 20,
        .score=0.5f,
        .nms=0.5f});
    //************************************************************
    cv::Mat frame;
    while (true) {
        if (!capture.read(frame) || frame.empty()) break;

        if(yolov10trace->inference(frame)){
            cv::imshow("frame", frame);
            int key = cv::waitKey(10);
            if (key == 'q' || key == 27) break;
        }else{
            std::println("inference error!!!");
            continue;
        }
    }
    capture.release();
}   


void dino(){
    auto dino = std::make_unique<GroundDino>();
    
    std::string onnx_path = "../models/grounded.onnx";
    dino->initialize(onnx_path,true);
    dino->setparms({.score=0.5f,.nms=0.8f,.prompt="biscuits"});
    std::string folder_path = "../assets/images/*.png";
    std::string output_path = "../assets/output/";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto& path : paths) {
        std::println("path={}",path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        int r = dino->inference(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("duration = {}ms",duration);
        if(r){
            auto filename = std::filesystem::path(path).filename().string();
            // cv::imwrite(output_path+filename,image);
            cv::imshow("Image", image);
            cv::waitKey(0);
        }else{
            std::println("inference error!!!");
            continue;
        }
    }
}
int main(int argc, char const *argv[]){
    // yolo();
    // yolosam();
    // yolotrace();
    dino();
    return 0;   
}

