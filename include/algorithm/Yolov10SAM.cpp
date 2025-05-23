#include "Yolov10SAM.h"


void Yolov10SAM::postprocess(std::vector<Ort::Value> &output_tensors){
    auto rand = [](uint _min, uint _max){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(_min, _max);
        return dis(gen);
    };
    cv::Mat output_mask(this->ori_img->size(),CV_8UC3,cv::Scalar(0,0,0));
    //并行求解所有通道的图像，保留合乎形状的通道
    tbb::parallel_for(tbb::blocked_range<int>(0, output_tensors.size(), 1),[&](const tbb::blocked_range<int>& r){
        auto index = r.begin();
        float* output = output_tensors[index].GetTensorMutableData<float>();
        cv::Mat outimg(ori_img->size(),CV_32FC1,output); // 这是padding图像上的输出mask
        cv::Mat dst;
        outimg.convertTo(dst, CV_8UC1, 255);
        cv::threshold(dst,dst,0,255,cv::THRESH_BINARY);
        //开运算
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);
        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(*ori_img, contours, -1, cv::Scalar(255,20,20),2,cv::LINE_AA);
        //*****************************************************
        std::vector<cv::Mat> subimg(3);
        int blue = rand(0,255),green = rand(0,255),red = rand(0,255);
        cv::threshold(dst,subimg[0],0,std::clamp(blue,0,70),cv::THRESH_BINARY);
        cv::threshold(dst,subimg[1],0,green,cv::THRESH_BINARY);
        cv::threshold(dst,subimg[2],0,red,cv::THRESH_BINARY);
        // 合并
        cv::Mat merge;
        cv::merge(subimg,merge);
        output_mask +=merge;
    });
    cv::cvtColor(output_mask,output_mask,cv::COLOR_BGR2BGRA);
    cv::cvtColor(*ori_img,*ori_img,cv::COLOR_BGR2BGRA);
    cv::addWeighted(output_mask,0.65,*ori_img,1,0,*ori_img);
}

int Yolov10SAM::setparms(ParamsSam parms){
    this->params = std::move(parms);
    return 1;
}

std::variant<bool,std::string> Yolov10SAM::initialize(std::vector<std::string>& onnx_paths, bool is_cuda){
    // 约定顺序是 yolov10.onnx,encoder.onnx,decoder.onnx
    assert(onnx_paths.size() == 3);
    auto is_file = [](const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
    };
    for (const auto& path : onnx_paths) {
        if (!is_file(path)) {
            return std::format("Model file dose not exist.file:{}",path);
        }
    }
    this->yolo_options.SetIntraOpNumThreads(2); //设置线程数量
    this->encoder_options.SetIntraOpNumThreads(2); //设置线程数量
    this->decoder_options.SetIntraOpNumThreads(2); //设置线程数量
    //***********************************************************
     if(is_cuda) {
        try {
            OrtCUDAProviderOptions options;
            options.device_id = 0;
            options.arena_extend_strategy = 0;
            options.gpu_mem_limit = SIZE_MAX;
            options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchHeuristic;
            options.do_copy_in_default_stream = 1;
            this->yolo_options.AppendExecutionProvider_CUDA(options);
            this->encoder_options.AppendExecutionProvider_CUDA(options);
            this->decoder_options.AppendExecutionProvider_CUDA(options);
            std::println("Using CUDA...");
        }catch (const std::exception& e) {
            std::string error(e.what());
            return error;
        }
    }else {
        std::println("Using CPU...");
    }
    //**************************************************************
    try {
#ifdef _WIN32
        unsigned len_yolo = onnx_paths[0].size() * 2; // 预留字节数
        unsigned len_encoder = onnx_paths[1].size() * 2; // 预留字节数
        unsigned len_decoder = onnx_paths[2].size() * 2; // 预留字节数
        setlocale(LC_CTYPE, ""); //必须调用此函数,本地化

        wchar_t* p_yolo = new wchar_t[len_yolo]; // 申请一段内存存放转换后的字符串
        wchar_t* p_encoder = new wchar_t[len_encoder]; // 申请一段内存存放转换后的字符串
        wchar_t* p_decoder = new wchar_t[len_decoder]; // 申请一段内存存放转换后的字符串
        mbstowcs(p_yolo, onnx_paths[0].c_str(), len_yolo); // 转换
        mbstowcs(p_encoder, onnx_paths[1].c_str(), len_encoder); // 转换
        mbstowcs(p_decoder, onnx_paths[2].c_str(), len_decoder); // 转换

        std::wstring wstr_yolo(p_yolo);
        std::wstring wstr_encoder(p_encoder);
        std::wstring wstr_decoder(p_decoder);
        delete[] p_yolo,p_encoder,p_decoder; // 释放申请的内存
        yolo_session = new Ort::Session(yolo_env, wstr_yolo.c_str(), this->yolo_options);
        encoder_session = new Ort::Session(encoder_env, wstr_encoder.c_str(), this->encoder_options);
        decoder_session = new Ort::Session(decoder_env, wstr_decoder.c_str(), this->decoder_options);
#else
        yolo_session = new Ort::Session(yolo_env, (const char*)onnx_paths[0].c_str(), this->yolo_options);
        encoder_session = new Ort::Session(encoder_env, (const char*)onnx_paths[1].c_str(), this->encoder_options);
        decoder_session = new Ort::Session(decoder_env, (const char*)onnx_paths[2].c_str(), this->decoder_options);
#endif
    }catch (const std::exception& e) {
        return std::format("Failed to load model. Please check your onnx file!");
    }
    //**************************************************************
    this->load_onnx_info(this->yolo_session,this->yolo_input_nodes,this->yolo_output_nodes);
    this->load_onnx_info(this->encoder_session,this->encoder_input_nodes,this->encoder_output_nodes);
    this->load_onnx_info(this->decoder_session,this->decoder_input_nodes,this->decoder_output_nodes);
    
    this->is_inited = true;
    std::println("initialize ok!!");
    return true;
}

void Yolov10SAM::preprocess(cv::Mat &image){
    cv::Mat image_ = image.clone();
    auto net_w = (float)this->yolo_input_nodes[0].dim.at(3);  
    auto net_h = (float)this->yolo_input_nodes[0].dim.at(2);  
    float scale = std::min(net_w/image.cols,net_h/image.rows);
    cv::resize(image_,image_,cv::Size(int(image.cols*scale),int(image.rows*scale)));
    int top     = (net_h - image_.rows) / 2;
    int bottom  = (net_h - image_.rows) / 2 + int(net_h - image_.rows) % 2;
    int left    = (net_w - image_.cols) / 2;
    int right   = (net_w - image_.cols) / 2 + int(net_w - image_.cols) % 2; 
    cv::copyMakeBorder(image_, image_, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114)); // padding
    input_images.clear();
    std::vector<cv::Mat> mats_{image_};
    cv::Mat blob_yolo = cv::dnn::blobFromImages(mats_, 1/255.0,cv::Size(net_w, net_h), cv::Scalar(0, 0, 0), true, false);
    
    cv::Mat image_sam;
    cv::subtract(image, cv::Scalar(0.406, 0.456, 0.485), image_sam);
    cv::divide(image_sam, cv::Scalar(0.225, 0.224, 0.229), image_sam);
    std::vector<cv::Mat> mats{image_sam};
    cv::Mat blob_encoder = cv::dnn::blobFromImages(mats, 1/255.0,image.size(), cv::Scalar(0, 0, 0), true, false);

    input_images.push_back(std::move(blob_yolo));
    input_images.push_back(std::move(blob_encoder));
}


std::variant<bool,std::string> Yolov10SAM::inference(cv::Mat &image){
    if (image.empty() || !is_inited) return "image can not empyt!";
    this->ori_img = &image;

    // 图片预处理
    try {
        this->preprocess(image); 
     }catch (const std::exception& e) {
        return "Image preprocess failed!";
    }
    // ******************************yolo推理**************************************
    std::vector<Ort::Value> yolo_input_tensor;
    yolo_input_tensor.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        this->input_images[0].ptr<float>(),
                        this->input_images[0].total(),
                        this->yolo_input_nodes[0].dim.data(),
                        this->yolo_input_nodes[0].dim.size())
    );
    std::vector<cv::Rect> boxes = this->yolo_infer(yolo_input_tensor);
    if(boxes.empty()) return "yolo can not detect any bbox!";
    // return 1;
    //*******************************encoder推理***********************************
    this->encoder_input_nodes[0].dim = {1,3,image.rows,image.cols};
    std::vector<Ort::Value> encoder_input_tensor;
    encoder_input_tensor.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        this->input_images[1].ptr<float>(),
                        this->input_images[1].total(),
                        this->encoder_input_nodes[0].dim.data(), // 3*1024*1024
                        this->encoder_input_nodes[0].dim.size())
    );
    Ort::Value image_embeddings = std::move(this->encoder_infer(encoder_input_tensor).at(0));
    //*******************************decoder推理***********************************
    std::vector<std::vector<float>> points_cord;
    for(auto &box:boxes){
        std::vector<float> point_val{(float)box.x,(float)box.y,(float)box.x+box.width,(float)box.y+box.height};//xyxy
        points_cord.emplace_back(point_val);
    }
    // 创建一个占位batched_point_coords,实际并不起作用
    std::vector<float>point_val{(float)boxes[0].x,(float)boxes[0].y,(float)boxes[0].x+boxes[0].width,(float)boxes[0].y+boxes[0].height};//xyxy
    auto batched_point_coords = Ort::Value::CreateTensor<float>(
                        memory_info,
                        point_val.data(),
                        point_val.size(),
                        this->decoder_input_nodes[1].dim.data(),
                        this->decoder_input_nodes[1].dim.size());
    // 添加batched_point_labels
    std::vector<float> point_labels = {2,3};
    auto batched_point_labels = Ort::Value::CreateTensor<float>(
                        memory_info,
                        point_labels.data(),
                        point_labels.size(),
                        this->decoder_input_nodes[2].dim.data(),
                        this->decoder_input_nodes[2].dim.size());
    // 添加orig_im_size
    std::vector<int64> img_size{image.rows,image.cols};//h,w
    auto orig_im_size = Ort::Value::CreateTensor<int64>(
                        memory_info,
                        img_size.data(),
                        img_size.size(),
                        this->decoder_input_nodes[3].dim.data(),
                        this->decoder_input_nodes[3].dim.size());

    std::vector<Ort::Value> output_tensors;
    std::vector<Ort::Value> input_tensors;
    // 移交所有权
    input_tensors.emplace_back(std::move(image_embeddings));
    input_tensors.emplace_back(std::move(batched_point_coords));
    input_tensors.emplace_back(std::move(batched_point_labels));
    input_tensors.emplace_back(std::move(orig_im_size));
    try {
        for(auto &points:points_cord){
            input_tensors[1] = Ort::Value::CreateTensor<float>(
                                memory_info,
                                points.data(),
                                points.size(),
                                this->decoder_input_nodes[1].dim.data(),
                                this->decoder_input_nodes[1].dim.size());
            output_tensors.emplace_back(std::move(this->decoder_infer(input_tensors).at(0)));
        }
    }catch (const std::exception& e) {
        return "decoder_infer  failed!!";
    }
    //***********************输出后处理**************************************
    try {
        this->postprocess(output_tensors);
    }catch (const std::exception& e) {
        return "tensor postprocess failed!!";
    }
    return true;
}
std::vector<cv::Rect> Yolov10SAM::yolo_infer(std::vector<Ort::Value> &input_tensor){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->yolo_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->yolo_output_nodes) output_names.push_back(node.name);
    //*******************************推理******************************
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = this->yolo_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),  // images
            input_tensor.data(), // 1*3*1024*1024
            input_tensor.size(), // 1
            output_names.data(), // output0
            output_names.size()); // 1
    }catch (const std::exception& e) {
        std::println("forward yolo model failed!!");
    }
    //*******************************后处理******************************
    float* output = output_tensors[0].GetTensorMutableData<float>(); // 1*300*6
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<cv::Rect> boxes;
    //******************************************************************
    auto net_w = (float)this->yolo_input_nodes[0].dim.at(3); // 1024
    auto net_h = (float)this->yolo_input_nodes[0].dim.at(2); // 1024
    float scale = std::min(net_w/ori_img->cols,net_h/ori_img->rows);

    for(size_t index = 0;index < this->yolo_output_nodes[0].dim[1]; index +=this->yolo_output_nodes[0].dim[2]){
        auto x1 = output[index];
        auto y1 = output[index + 1] ;
        auto x2  = output[index + 2];
        auto y2  = output[index + 3];
        auto score = output[index + 4];
        auto label = (int)output[index + 5];

        auto pad_w = (net_w - ori_img->cols * scale) / 2;
        auto pad_h = (net_h - ori_img->rows * scale) / 2;

        x1 = (x1 - pad_w) / scale;
        x2 = (x2 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
        y2 = (y2 - pad_h) / scale;
        cv::Rect rect(cv::Point2f(x1,y1),cv::Size2f(x2-x1,y2-y1));
        boxes.emplace_back(rect);
        labels.emplace_back(label);
        scores.emplace_back(score);
    }
    //*****************************************************
    std::vector<int>indices;
    cv::dnn::NMSBoxes(boxes,scores,this->params.score,this->params.nms,indices);
   
    // draw boxes
    std::vector<cv::Rect> result;
    for(const auto i:indices){
        std::string name = LABEL.at(labels[i]);
        std::size_t hash = std::hash<std::string>{}(name);
        double r = (hash & 0xFF0000) >> 16;
        double g = (hash & 0x00FF00) >> 8;
        double b = hash & 0x0000FF;
        ///**************************************************
        cv::rectangle(*ori_img,boxes[i],cv::Scalar{b,g,r},1);
        cv::putText(*ori_img,std::format("{}:{:.2f}",name,scores[i]),cv::Point{boxes[i].x,boxes[i].y-5}
        ,1,1.1,cv::Scalar{b,g,r});
        //**************************************************
        cv::Rect temp = boxes[i];
        result.push_back(std::move(temp));
    }
    return std::move(result);
}

std::vector<Ort::Value> Yolov10SAM::encoder_infer(std::vector<Ort::Value> &input_tensor){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->encoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->encoder_output_nodes) output_names.push_back(node.name);
    //*******************************推理******************************
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = this->encoder_session->Run(
            Ort::RunOptions{nullptr}, //默认
            input_names.data(),     //输入节点的所有名字
            input_tensor.data(),    //输入tensor
            input_tensor.size(),    //输入tensor的数量
            output_names.data(),    //输出节点的所有名字
            output_names.size()     //输出节点名字的数量
        );
    }catch (const std::exception& e) {
        std::println("forward encoder model failed!!");
    }
    return std::move(output_tensors);
}
std::vector<Ort::Value> Yolov10SAM::decoder_infer(std::vector<Ort::Value>& input_tensor){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->decoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->decoder_output_nodes) output_names.push_back(node.name);
    //*******************************推理******************************
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = this->decoder_session->Run(
            Ort::RunOptions{nullptr}, //默认
            input_names.data(),     //输入节点的所有名字
            input_tensor.data(),    //输入tensor
            input_tensor.size(),    //输入tensor的数量
            output_names.data(),    //输出节点的所有名字
            output_names.size()     //输出节点名字的数量
        );
    }catch (const std::exception& e) {
        std::println("forward decoder model failed!!");
    }
    return std::move(output_tensors);
}

