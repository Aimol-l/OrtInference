#include "PaddleOCR.h"


std::variant<bool,std::string> PaddleOCR::initialize(std::vector<std::string>& onnx_paths, bool is_cuda){

    assert(std::ssize(onnx_paths) == 3);
    if (!MT::FileExists(onnx_paths[0])) {
        auto info = std::format("Model file dose not exist.file:{}", onnx_paths[0]);
        return info;
    }
    if (!MT::FileExists(onnx_paths[1])) {
        auto info = std::format("Model file dose not exist.file:{}", onnx_paths[1]);
        return info;
    }
    if (!MT::FileExists(onnx_paths[2])) {
        auto info = std::format("Model file dose not exist.file:{}", onnx_paths[2]);
        return info;
    }
    auto env_name_det = std::format("ocr_det");
    auto env_name_cls = std::format("ocr_cls");
    auto env_name_rec = std::format("ocr_rec");
    this->env_det = Ort::Env(ORT_LOGGING_LEVEL_WARNING,env_name_det.c_str());
    this->env_cls = Ort::Env(ORT_LOGGING_LEVEL_WARNING,env_name_cls.c_str());
    this->env_rec = Ort::Env(ORT_LOGGING_LEVEL_WARNING,env_name_rec.c_str());

    this->options_det.SetIntraOpNumThreads(2); //设置线程数量
    this->options_det.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    this->options_cls.SetIntraOpNumThreads(2); //设置线程数量
    this->options_cls.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    this->options_rec.SetIntraOpNumThreads(2); //设置线程数量
    this->options_rec.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    if (is_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = SIZE_MAX;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchHeuristic;
            cuda_options.do_copy_in_default_stream = 1;
            this->options_det.AppendExecutionProvider_CUDA(cuda_options);
            this->options_cls.AppendExecutionProvider_CUDA(cuda_options);
            this->options_rec.AppendExecutionProvider_CUDA(cuda_options);
            std::println("Using CUDA...");

            // const auto& api = Ort::GetApi();
            // OrtTensorRTProviderOptionsV2* tensorrt_options;
            // Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
            // std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
            // Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(this->session_options),rel_trt_options.get()));
            // std::println("Using TensorRT...");
        }catch (const std::exception& e) {
            std::string error(e.what());
            return error;
        }
    }else {
        std::println("Using CPU...");
    }
    try {
#ifdef _WIN32
    unsigned len_det = onnx_paths[0].size() * 2; // 预留字节数
    unsigned len_cls = onnx_paths[1].size() * 2; // 预留字节数
    unsigned len_rec = onnx_paths[2].size() * 2; // 预留字节数

    setlocale(LC_CTYPE, ""); //必须调用此函数,本地化

    wchar_t* p_det = new wchar_t[len_det]; // 申请一段内存存放转换后的字符串
    wchar_t* p_cls = new wchar_t[len_cls];
    wchar_t* p_rec = new wchar_t[len_rec];

    mbstowcs(p_det, onnx_paths[0].c_str(), len_det); // 转换
    mbstowcs(p_cls, onnx_paths[1].c_str(), len_cls);
    mbstowcs(p_rec, onnx_paths[2].c_str(), len_rec);

    std::wstring wstr_det(p_det);
    std::wstring wstr_cls(p_cls);
    std::wstring wstr_rec(p_rec);

    delete[] p_det,p_cls,p_rec; // 释放申请的内存
    this->session_det = new Ort::Session(env_det, wstr_det.c_str(), this->options_det);
    this->session_cls = new Ort::Session(env_cls, wstr_cls.c_str(), this->options_cls);
    this->session_rec = new Ort::Session(env_rec, wstr_rec.c_str(), this->options_rec);
#else
    this->session_det = new Ort::Session(env_det, (const char*)onnx_paths[0].c_str(), this->options_det);
    this->session_cls = new Ort::Session(env_cls, (const char*)onnx_paths[1].c_str(), this->options_cls);
    this->session_rec = new Ort::Session(env_rec, (const char*)onnx_paths[2].c_str(), this->options_rec);
#endif
    }catch (const std::exception& e) {
        return std::format("Failed to load model. Please check your onnx file!");
    }
    this->load_onnx_info(this->session_det,this->input_nodes_det,this->output_nodes_det,"det.onnx");
    this->load_onnx_info(this->session_cls,this->input_nodes_cls,this->output_nodes_cls,"cls.onnx");
    this->load_onnx_info(this->session_rec,this->input_nodes_rec,this->output_nodes_rec,"rec.onnx");
    
    this->is_inited = true;
    // std::println("initialize ok!!");
    return true;
}
int PaddleOCR::setparms(ParamsOCR parms){
    this->params = std::move(parms);
    if (!MT::FileExists(parms.dictionary)) {
        std::println("dictionary file not exist:{}",parms.dictionary);
        return 0;
    }else{
        std::string dict_path(parms.dictionary);
        if(this->dictionary.size() <=0) this->dictionary.load(dict_path);
    }
    return 1;
}
void PaddleOCR::preprocess(cv::Mat &image){
    // 获取原始图像的宽度和高度
    int original_width = image.cols;
    int original_height = image.rows;
    // 计算最近的32的倍数
    int new_width = static_cast<int>(std::ceil(original_width / 32.0)) * 32;
    int new_height = static_cast<int>(std::ceil(original_height / 32.0)) * 32;
    // 调整图像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));
    // 减去mean,除以std
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
    cv::subtract(resized_image, cv::Scalar(0.485,0.456,0.406), resized_image);
    cv::divide(resized_image, cv::Scalar(0.229,0.224,0.225), resized_image);
    // 将图像转换为模型输入格式
    cv::Mat blob = cv::dnn::blobFromImage(resized_image, 1,cv::Size(new_width, new_height), cv::Scalar(0, 0, 0), true, false);
    this->input_nodes_det[0].dim = {1, 3, new_height, new_width};
    // 清空并保存处理后的图像
    input_images.clear();
    input_images.push_back(std::move(blob));
}
std::variant<bool,std::string> PaddleOCR::inference(cv::Mat &image){
    if (image.empty()) return "image can not empyt!";
    if (!this->is_inited) return "model not inited!";
    this->ori_img = &image;
    try {
        this->preprocess(image); 
     }catch (const std::exception& e) {
        return std::format("Image preprocess failed! {}",e.what());
    }
    std::optional<std::vector<cv::Mat>> det_result = this->infer_det(); 
    if (!det_result.has_value()) return "Detection failed! can't find any text!";
    std::println("Detection success!");

    std::optional<std::vector<cv::Mat>> cls_result = this->infer_cls(det_result.value()); // 返回的是预处理好的子图
    if(!cls_result.has_value()) return "Classification failed!";
    std::println("Classification success!");

    std::optional<std::vector<Ort::Value>> rec_result = this->infer_rec(cls_result.value());
    if(!rec_result.has_value()) return "Recognition failed!";
    std::println("Recognition success!");

    this->postprocess(rec_result.value());

    return true;
}
void PaddleOCR::postprocess(std::vector<Ort::Value> &output_tensors){
    float* output = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // [batch,M,15631],M是句子最大字符数量
    //std::println("output shape=[{},{},{}]",output_shape[0],output_shape[1],output_shape[2]); // [batch,92,15631]
    // [batch,M]
    std::vector<std::vector<float>> max_values(output_shape[0], std::vector<float>(output_shape[1], -std::numeric_limits<float>::infinity()));
    // [batch,M]
    std::vector<std::vector<size_t>> max_indices(output_shape[0], std::vector<size_t>(output_shape[1], -1));
    // 用tbb并行处理每个batch和每个M维
    tbb::parallel_for(0,int(output_shape[0]),1,[&](int batch){ // batch
        tbb::parallel_for(0,int(output_shape[1]),1,[&](int M){ // M
            int max_idx = -1;
            float max_val = -std::numeric_limits<float>::infinity();
            for(int i=0;i<output_shape[2];i++){ // 6625
                int index = batch * output_shape[1] * output_shape[2] + M * output_shape[2] + i;
                float value = output[index];
                if(value > max_val){
                    max_val = value;
                    max_idx = i;
                }
            }
            max_values[batch][M] = max_val;
            max_indices[batch][M] = max_idx;
        });
    });
    // 准备解码
    std::vector<std::string> results_text(output_shape[0]);
    // indix --> text
    for(size_t batch = 0;batch <output_shape[0];batch++){ // batch
        //去重
        std::vector<bool> selection(output_shape[1],true);
        for (size_t i = 1;i < output_shape[1];i++) selection[i] = (max_indices[batch][i] != max_indices[batch][i - 1]);
        for(size_t j = 0;j<output_shape[1];j++){ // 
            if (!selection[j] && this->params.repeat) continue;
            if(max_indices[batch][j] == 0  || max_values[batch][j] <this->params.text) continue;
            auto text = this->dictionary.get_char(max_indices[batch][j]-1);
            results_text[batch] += text;
        }
        // 绘制
        cv::RotatedRect rorect = cv::minAreaRect(this->polygons[batch].points);
        std::vector<cv::Point2f> ropoints;
        rorect.points(ropoints);
        cv::putText(*ori_img,std::to_string(batch),ropoints[0],cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(50,25,255),1);
        if(results_text[batch].empty()){
            for(int i =0;i<4;i++)   cv::line(*ori_img,ropoints[i],ropoints[(i+1)%4],cv::Scalar(50,0,255),1);
        }else{
            std::println("Text {} => {}",batch,results_text[batch]);
            for(int i =0;i<4;i++)   cv::line(*ori_img,ropoints[i],ropoints[(i+1)%4],cv::Scalar(25,250,50),1);
        }
    }
}
std::vector<cv::Point2f> PaddleOCR::unclip(std::vector<cv::Point> &polygon){
    double area = cv::contourArea(polygon);
    double length = cv::arcLength(polygon, true);
    double distance = area * this->params.unclip_ratio / length;
    
    Clipper2Lib::Path64 path;
    for (const auto& pt : polygon) {
        path.push_back(Clipper2Lib::Point64(pt.x, pt.y));
    }
    Clipper2Lib::ClipperOffset clipper;
    clipper.AddPath(path, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
    
    Clipper2Lib::Paths64 solution;
    clipper.Execute(distance, solution);  // offset 是扩展距离（单位：像素）
    std::vector<cv::Point2f> result;
    if (solution.empty()) {
        std::cerr << "ClipperOffset failed: solution is empty!" << std::endl;
        return result;
    }
    Clipper2Lib::Path64& expanded = solution[0];
    // 5. 转换回 OpenCV 的 Point
    for (const auto& pt : expanded) {
        result.push_back(cv::Point2f(pt.x,pt.y));
    }
    return result;
}
float PaddleOCR::box_score_slow(cv::Mat &pred, std::vector<cv::Point> &approx){
    // 边界检查
    if (pred.empty() || approx.empty()) {
        return 0.0f;
    }
    // 获取图像尺寸
    const int height = pred.rows;
    const int width = pred.cols;
    // 创建多边形掩码
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    // 填充多边形(转换为32位整数坐标)
    std::vector<cv::Point> int_points;
    for (const auto& pt : approx) {
        int_points.emplace_back(cv::Point(
            std::clamp(pt.x, 0, width - 1),
            std::clamp(pt.y, 0, height - 1)
        ));
    }
    std::vector<std::vector<cv::Point>> contours = {int_points};
    cv::fillPoly(mask, contours, cv::Scalar(1));
    // 计算非零像素数量（可选调试信息）
    // int pixel_count = cv::countNonZero(mask);
    // if (pixel_count == 0) return 0.0f;
    // 计算均值并返回
    return static_cast<float>(cv::mean(pred, mask)[0]);
}

std::vector<PaddleOCR::Polygon> PaddleOCR::poly_from_bitmap(cv::Mat &pred, cv::Mat &bitmap){
    std::vector<PaddleOCR::Polygon> result;
    cv::Mat binmat;
    bitmap.convertTo(binmat, CV_8UC1,255);
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binmat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for(auto &contour:contours){
        // 过滤太小的轮廓
        double temp_area = cv::contourArea(contour);
        if(temp_area < this->params.min_area) continue;
        float epsilon = 0.001 * cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx,epsilon, true); // 简化轮廓
        if(approx.size() < 4) continue;
        float score = this->box_score_slow(pred, approx);
        if(score < this->params.thresh)  {
            // std::println("Skipping contour with score {}", score);
            continue;
        }
        // 边界扩展
        auto unclip_poly = this->unclip(approx);
        float scale_x = this->ori_img->cols / static_cast<float>(this->input_nodes_det[0].dim[3]);
        float scale_y = this->ori_img->rows / static_cast<float>(this->input_nodes_det[0].dim[2]);
        std::transform(unclip_poly.begin(), unclip_poly.end(), unclip_poly.begin(),
            [&](const cv::Point2f& pt) {
                // std::println("[{:.2f},{:.2f}] -> [{:.2f},{:.2f}]",pt.x,pt.y,pt.x*scale_x,pt.y*scale_y);
                return cv::Point2f(pt.x * scale_x, pt.y * scale_y);
            });
        result.push_back({score,unclip_poly});
    }
    return result;
}
std::optional<std::vector<cv::Mat>> PaddleOCR::infer_det(){
    // 创建输入张量
    std::vector<Ort::Value> input_tensor;
    try {
        input_tensor.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            this->input_images[0].ptr<float>(),
            this->input_images[0].total(),
            this->input_nodes_det[0].dim.data(),
            this->input_nodes_det[0].dim.size())
        );
    }catch (const std::exception& e) {
        return std::nullopt; // 无值
    }
    // 模型推理
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->input_nodes_det)  input_names.push_back(node.name);
    for(auto &node:this->output_nodes_det) output_names.push_back(node.name);
    std::vector<Ort::Value> output_tensor; // [1,3,h,w] --> [1,1,h,w]
    try {
        output_tensor = this->session_det->Run(
            Ort::RunOptions{ nullptr }, //默认
            input_names.data(), 
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()
        );
    }catch (const std::exception& e) {
        return std::nullopt; // 无值
    }
    // 后处理mask: 
    float* output = output_tensor[0].GetTensorMutableData<float>();
    auto output_shape = output_tensor[0].GetTensorTypeAndShapeInfo().GetShape(); // [1,1,h,w]
    std::println("det output shape=[{},{},{},{}]",output_shape[0],output_shape[1],output_shape[2],output_shape[3]); // 
    cv::Mat prob = cv::Mat((size_t)output_shape[2],(size_t)output_shape[3],CV_32FC1,output); // 概率图
    // cv::imshow("prob",prob);
    // cv::waitKey(0);
    // 将prob中小于thresh的置为0
    cv::Mat bitmap;
    cv::threshold(prob,bitmap,this->params.thresh,1.0,cv::THRESH_BINARY);
    cv::rectangle(bitmap,cv::Point(0,0),cv::Point(bitmap.cols-1,bitmap.rows-1),cv::Scalar(0),2);
    // 结合概率图和分割图计算文本框位置
    this->polygons = this->poly_from_bitmap(prob,bitmap); // 坐标是在ori_img上的
    if(polygons.empty()) {
        return std::nullopt;
    }
    auto point_dis = [&](cv::Point2f p1,cv::Point2f p2){
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    };
    auto get_angle = [&](cv::Point2f p1,cv::Point2f p2){
        return std::atan2(p2.y - p1.y, p2.x - p1.x)* 180.0 / CV_PI;
    };
    std::vector<cv::Mat> images;
    int input_h = this->input_nodes_rec[0].dim[2]; // 48 or 64 or 80
    for(auto &poly:polygons){ 
        cv::RotatedRect rorect = cv::minAreaRect(poly.points);
        std::vector<cv::Point2f> ropoints;
        rorect.points(ropoints);
        auto dis1 = point_dis(ropoints[0], ropoints[1]);
        auto dis2 = point_dis(ropoints[1], ropoints[2]);
        if(dis1>dis2){
            std::swap(rorect.size.width,rorect.size.height);
            rorect.angle = get_angle(ropoints[0], ropoints[1]);
        }else{
            rorect.angle = get_angle(ropoints[1], ropoints[2]);
        }
        // std::println("rorect:[{},{},{},{}] {}",rorect.center.x,rorect.center.y,rorect.size.width,rorect.size.height,rorect.angle);
        // 裁剪图片并旋转正常
        cv::Rect rect = cv::boundingRect(poly.points);
        if(rect.x<0) rect.x = 0;
        if(rect.y<0) rect.y = 0;
        if(rect.x + rect.width > (*ori_img).cols) 
            rect.width = (*ori_img).cols - rect.x;
        if(rect.y + rect.height > (*ori_img).rows) 
            rect.height = (*ori_img).rows - rect.y;
        cv::Mat crop_img = (*ori_img)(rect).clone();
        // for(int i=0;i<4;i++) cv::line(bitmap,ropoints[i],ropoints[(i+1)%4],cv::Scalar(128),2);
        // cv::imshow("bitmap",bitmap);
        // cv::imshow("crop_img",crop_img);
        // cv::waitKey(0);
        int new_w = std::max(crop_img.cols,crop_img.rows)*1.5;
        cv::Mat rotated_image(new_w,new_w, CV_8UC3,cv::Scalar(0,0,0));
        cv::Rect roi = {(new_w-crop_img.cols)/2,(new_w-crop_img.rows)/2,crop_img.cols,crop_img.rows};
        crop_img.copyTo(rotated_image(roi));
        rorect.center.x = int(rotated_image.cols / 2);
        rorect.center.y = int(rotated_image.rows / 2);

        cv::Mat rotation_matrix = cv::getRotationMatrix2D(rorect.center, rorect.angle,1.0);
        cv::warpAffine(rotated_image, rotated_image, rotation_matrix, rotated_image.size(), cv::INTER_CUBIC);
        cv::Mat cropped_image;
        cv::getRectSubPix(rotated_image, rorect.size, rorect.center, cropped_image);
        cv::Size corp_size(int(input_h*rorect.size.width/rorect.size.height),input_h);
        cv::resize(cropped_image, cropped_image,corp_size);
        // cv::imshow("cropped_image",cropped_image);
        // cv::waitKey(0);
        images.push_back(cropped_image);
    }
    if(images.empty()) return std::nullopt;
    // cv::imshow("bitmap",bitmap);
    // cv::waitKey(0);
    std::reverse(images.begin(), images.end());
    std::reverse(polygons.begin(), polygons.end());
    return images;
}
std::optional<std::vector<cv::Mat>> PaddleOCR::infer_cls(std::vector<cv::Mat> &images){
    int input_h = this->input_nodes_rec[0].dim[2]; // 
    int max_w = 320;    // 确保宽度是32的倍数,最好是320
    for (const auto& img : images) {
        int w = img.cols;
        int h = img.rows;
        float ratio = static_cast<float>(w) / h;
        max_w = std::max(max_w, static_cast<int>(std::ceil(input_h * ratio / 32.0) * 32));
    }
    std::vector<cv::Mat> norm_images(images.size());
    std::vector<cv::Mat> cls_images(images.size());
    int cls_w = this->input_nodes_cls[0].dim[3] > 0 ? this->input_nodes_cls[0].dim[3]:192;  //旧的模型是192,新的是160
    int cls_h = this->input_nodes_cls[0].dim[2] > 0 ? this->input_nodes_cls[0].dim[2]:48;   //旧的模型是48,新的是80
    tbb::parallel_for(0,int(images.size()),1, [&](int idx){
        // det w = 320
        cv::Mat padded_img = MT::PaddingImg(images[idx],cv::Size(max_w,input_h));
        padded_img.convertTo(padded_img, CV_32FC3,1/255.0);
        cv::subtract(padded_img, cv::Scalar(0.5,0.5,0.5), padded_img);
        cv::divide(padded_img, cv::Scalar(0.5,0.5,0.5), padded_img);
        norm_images[idx] = padded_img;
        // cls w = 160
        cv::Mat cls_img = MT::PaddingImg(images[idx],cv::Size(cls_w,cls_h));
        images[idx].convertTo(cls_img, CV_32FC1,1/255.0);
        cv::subtract(cls_img, cv::Scalar(0.5,0.5,0.5), cls_img);
        cv::divide(cls_img, cv::Scalar(0.5,0.5,0.5), cls_img);
        cls_images[idx] = cls_img;
    });
    // 判断图片的方向性
    cv::Mat blob = cv::dnn::blobFromImages(cls_images, 1,cv::Size(cls_w,cls_h),cv::Scalar(0,0,0),true,false);
    std::vector<Ort::Value> input_tensor;
    this->input_nodes_cls[0].dim = {(int64_t)images.size(),3,cls_h,cls_w};
    try {
        input_tensor.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            blob.ptr<float>(),
            blob.total(),
            this->input_nodes_cls[0].dim.data(),
            this->input_nodes_cls[0].dim.size())
        );
    }catch (const std::exception& e) {
        std::println("creat tensor failed:{}", e.what());
        return std::nullopt; // 无值
    }
    // 模型推理
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->input_nodes_cls)  input_names.push_back(node.name);
    for(auto &node:this->output_nodes_cls) output_names.push_back(node.name);
    std::vector<Ort::Value> output_tensor; // [batch,3,h,w] --> [batch,2]
    try {
        output_tensor = this->session_cls->Run(
            Ort::RunOptions{ nullptr }, //默认
            input_names.data(), 
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()
        );
    }catch (const std::exception& e) {
        std::println("{}", e.what());
        return std::nullopt; // 无值
    }
    // 获取输出数据
    float* output = output_tensor[0].GetTensorMutableData<float>();
    auto output_shape = output_tensor[0].GetTensorTypeAndShapeInfo().GetShape(); // [batch,2] // {0,180}

    // 最好统计一个正反文字的比例，防止误判。如果输出一张图片上绝大多数都是正文字，那么判断出的反文字大概率是误判了
    for(int i = 0; i < output_shape[0]; i++){
       int pos = i * output_shape[1];
        if(output[pos] < output[pos+1]){
            // std::println("[{:.2f},{:.2f}]",output[pos],output[pos+1]);
            cv::rotate(norm_images[i], norm_images[i], cv::ROTATE_180);// norm_images[i]旋转180度
            // cv::imshow("cls", cls_images[i]);
            // cv::waitKey(0);
        }
    }
    return norm_images; // 图片大小统一，文字方向也统一
}

std::optional<std::vector<Ort::Value>> PaddleOCR::infer_rec(std::vector<cv::Mat> &images){
    cv::Mat blob = cv::dnn::blobFromImages(images,1,images[0].size(),cv::Scalar(0,0,0),true,false);
    // 创建tensor
    std::vector<Ort::Value> input_tensor;
    this->input_nodes_rec[0].dim = { (int64_t)images.size(),3,images[0].rows,images[0].cols};
    try {
        input_tensor.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            blob.ptr<float>(),
            blob.total(),
            this->input_nodes_rec[0].dim.data(),
            this->input_nodes_rec[0].dim.size())
        );
    }catch (const std::exception& e) {
        std::println("{}", e.what());
        return std::nullopt; // 无值
    }
    // 模型推理
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->input_nodes_rec)  input_names.push_back(node.name);
    for(auto &node:this->output_nodes_rec) output_names.push_back(node.name);
    std::vector<Ort::Value> output_tensor; // [batch,3,48,W] --> [batch,32,6625]
    try {
        output_tensor = this->session_rec->Run(
            Ort::RunOptions{ nullptr }, //默认
            input_names.data(), 
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()
        );
    }catch (const std::exception& e) {
        std::println("{}", e.what());
        return std::nullopt; // 无值
    }
    return output_tensor;
}

