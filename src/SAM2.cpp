#include "SAM2.h"

std::variant<bool,std::string> SAM2::initialize(std::vector<std::string>& onnx_paths, bool is_cuda){
    // image_encoder,memory_attention,image_encoder,memory_encoder
    assert(onnx_paths.size() == 4);
    auto is_file = [](const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
    };
    for (const auto& path : onnx_paths) {
        if (!is_file(path))
            return std::format("Model file dose not exist.file:{}",path);
    }
    this->img_encoder_options.SetIntraOpNumThreads(2); //设置线程数量
    this->img_decoder_options.SetIntraOpNumThreads(2); //设置线程数量
    this->mem_attention_options.SetIntraOpNumThreads(2); //设置线程数量
    this->mem_encoder_options.SetIntraOpNumThreads(2); //设置线程数量
    //***********************************************************
    if(is_cuda) {
        try {
            OrtCUDAProviderOptions options;
            options.device_id = 0;
            options.arena_extend_strategy = 0;
            options.gpu_mem_limit = SIZE_MAX;
            options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            options.do_copy_in_default_stream = 1;
            this->img_encoder_options.AppendExecutionProvider_CUDA(options);
            this->img_decoder_options.AppendExecutionProvider_CUDA(options);
            this->mem_attention_options.AppendExecutionProvider_CUDA(options);
            this->mem_encoder_options.AppendExecutionProvider_CUDA(options);
            std::println("Using CUDA...");
        }catch (const std::exception& e) {
            std::string error(e.what());
            return error;
        }
    }else {
        std::println("Using CPU...");
    }
 try {
#ifdef _WIN32
    // todo
    return -4;
#else
        img_encoder_session = new Ort::Session(img_encoder_env, (const char*)onnx_paths[0].c_str(), this->img_encoder_options);
        mem_attention_session = new Ort::Session(mem_attention_env, (const char*)onnx_paths[1].c_str(), this->mem_attention_options);
        img_decoder_session = new Ort::Session(img_decoder_env, (const char*)onnx_paths[2].c_str(), this->img_decoder_options);
        mem_encoder_session = new Ort::Session(mem_encoder_env, (const char*)onnx_paths[3].c_str(), this->mem_encoder_options);
#endif  
    }catch (const std::exception& e) {
        return std::format("Failed to load model. Please check your onnx file!");
    }
    //************************************************************
    Ort::AllocatorWithDefaultOptions allocator;
    size_t const img_encoder_input_num = this->img_encoder_session->GetInputCount();
    size_t const img_decoder_input_num = this->img_decoder_session->GetInputCount();
    size_t const mem_attention_input_num = this->mem_attention_session->GetInputCount();
    size_t const mem_encoder_input_num = this->mem_encoder_session->GetInputCount();
    std::println("[{},{},{},{}]",img_encoder_input_num,img_decoder_input_num,mem_attention_input_num,mem_encoder_input_num);
    //************************************************************
    for (size_t index = 0; index < img_encoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->img_encoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->img_encoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_encoder_input_nodes.push_back(node);
    }
    for (size_t index = 0; index < img_decoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->img_decoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->img_decoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_decoder_input_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_attention_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->mem_attention_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->mem_attention_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_attention_input_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_encoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->mem_encoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->mem_encoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j=0;j<input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_encoder_input_nodes.push_back(node);
    }
    //************************************************************
    size_t const img_encoder_output_num = this->img_encoder_session->GetOutputCount();
    size_t const img_decoder_output_num = this->img_decoder_session->GetOutputCount();
    size_t const mem_attention_output_num = this->mem_attention_session->GetOutputCount();
    size_t const mem_encoder_output_num = this->mem_encoder_session->GetOutputCount();
    //************************************************************
    for (size_t index = 0; index < img_encoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->img_encoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->img_encoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_encoder_output_nodes.push_back(node);
    }
    for (size_t index = 0; index < img_decoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->img_decoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->img_decoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->img_decoder_output_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_attention_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->mem_attention_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->mem_attention_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_attention_output_nodes.push_back(node);
    }
    for (size_t index = 0; index < mem_encoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->mem_encoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->mem_encoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for(size_t j = 0;j<output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy(node.name, name, name_length);
        this->mem_encoder_output_nodes.push_back(node);
    }
    
    //************************************************************
    std::println("----------------img_encoder------------------");
    for(const auto& outputs: img_encoder_input_nodes){
        std::print("{}=[",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    std::println("-----------------img_decoder-----------------");
    for(const auto& outputs: img_decoder_input_nodes){
        std::print("{}=[",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    std::println("----------------mem_attention------------------");
    for(const auto& outputs: mem_attention_input_nodes){
        std::print("{}=[",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    std::println("----------------mem_encoder------------------");
    for(const auto& outputs: mem_encoder_input_nodes){
        std::print("{}=[",outputs.name);
        for(size_t i=0;i<outputs.dim.size()-1;i++)  std::print("{},",outputs.dim[i]);
        std::println("{}]",outputs.dim[outputs.dim.size()-1]);
    }
    std::println("----------------------------------");
    this->is_inited = true;
    std::println("initialize ok!!");
    return true;
}

std::variant<bool,std::string> SAM2::inference(cv::Mat &image){
    if (image.empty() || !is_inited) return "image can not empyt!";
    this->ori_img = &image;
    // 图片预处理
    try {
        this->preprocess(image); 
     }catch (const std::exception& e) {
        return "Image preprocess failed!";
    }
    // 图片编码器，输入图片
    std::vector<Ort::Value> img_encoder_tensor;
    img_encoder_tensor.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        this->input_images[0].ptr<float>(),
                        this->input_images[0].total(),
                        this->img_encoder_input_nodes[0].dim.data(),
                        this->img_encoder_input_nodes[0].dim.size())
    );
    auto result_0 = this->img_encoder_infer(img_encoder_tensor);
    if(result_0.index() != 0) return std::get<std::string>(result_0);

    auto result_1 =this->mem_attention_infer(); // 第0帧的时候没有进行推理
    if(result_1.index() != 0) return std::get<std::string>(result_1);

    auto result_2 =this->img_decoder_infer();
    if(result_2.index() != 0) return std::get<std::string>(result_2);

    auto result_3 =this->mem_encoder_infer();
    if(result_3.index() != 0) return std::get<std::string>(result_3);

    // 后处理
    std::vector<Ort::Value> output_tensors;
    output_tensors.push_back(std::move(this->img_decoder_out[2]));
    try {
        this->postprocess(output_tensors);
    }catch (const std::exception& e) {
        return "tensor postprocess failed!!";
    }
    this->infer_status.current_frame++;
    return true;
}
// [num_obj_ptr, current_vision_feat, current_vision_pos_embed, memory_0,memory_1, memory_pos_embed]->[image_embed]
std::vector<Ort::Value> SAM2::build_mem_attention_input(){
    std::vector<Ort::Value> input_tensor; // 5
    input_tensor.push_back(std::move(this->img_encoder_out[3])); //current_vision_feat
    input_tensor.push_back(std::move(this->img_encoder_out[4])); //current_vision_pos_embed
    // 需要构造出 memory 和 memory_pos_embed
    // memory是由obj_ptr_first，obj_ptr_recent和status_recent.maskmem_features 构造出的
    size_t obj_buffer_size = 1 + infer_status.obj_ptr_recent.size();//1+0,1+1,1+2,...
    std::vector<int64_t> dimensions_0{(int64_t)obj_buffer_size,256}; // [y,256]
    float* obj_ptrs = new float[obj_buffer_size*256]; // first+recent
    const float* tensor_data = infer_status.obj_ptr_first[0].GetTensorData<float>();
    memcpy(obj_ptrs, tensor_data, 256 * sizeof(float)); // 只复制了一部分

    for(size_t i = 0;i<infer_status.obj_ptr_recent.size();i++){
        auto& temp_tensor = infer_status.obj_ptr_recent.at(i);
        tensor_data = temp_tensor.GetTensorData<float>();
        memcpy(obj_ptrs+256*(i+1), tensor_data, 256 * sizeof(float));
    }
    auto memory_0 = Ort::Value::CreateTensor<float>(
                    memory_info,
                    obj_ptrs,
                    obj_buffer_size*256,
                    dimensions_0.data(),
                    dimensions_0.size()
                    );
    size_t features_size = infer_status.status_recent.size(); // 1,2,3,...,7
    float* maskmem_features_ = new float[features_size*64*64*64]; // 申请内存
    for(size_t i = 0;i<features_size;i++){
        auto& aaa = this->infer_status.status_recent.at(i);
        auto& temp_tensor = aaa.maskmem_features;
        tensor_data = temp_tensor[0].GetTensorData<float>();
        memcpy(maskmem_features_+64*64*64*i, tensor_data, 64*64*64*sizeof(float));
    }
    std::vector<int64_t> dimensions_1{(int64_t)features_size,64,64,64}; // [x,64,64,64]
    auto memory_1 = Ort::Value::CreateTensor<float>(
                    memory_info,
                    maskmem_features_,
                    features_size*64*64*64,
                    dimensions_1.data(),
                    dimensions_1.size()
                    );
    input_tensor.push_back(std::move(memory_0));
    input_tensor.push_back(std::move(memory_1));


    //***********************************************************************
    // memory_pos_embed是由两部分组成的。
    const float* temporal_code_ = this->mem_encoder_out[2].GetTensorData<float>();// [7,1,1,64]
    std::vector<const float*> temporal_code;
    for(int i = 6;i>=0;i--){
        auto temp = temporal_code_+i*64;
        temporal_code.push_back(temp);
    }
    size_t maskmem_buffer_size = infer_status.status_recent.size();
    size_t maskmem_pos_enc_size = (maskmem_buffer_size*4096+4*std::min(infer_status.current_frame,16))*64;
    
    float* maskmem_pos_enc_ = new float[maskmem_pos_enc_size];// 1049600
    
    // a[] , b[4096,1,64], c[1,1,64]
    auto tensor_add = [&](float* a,const float* b,const float* c){
        // b+c,结果保存到a
        for(int i =0;i<4096;i++){
            for(int j =0;j<64;j++){
                a[i*64+j] = b[i*64+j] + c[j];
            }
        }
    };
    // 第一部分：
    for(size_t j = 0;j<maskmem_buffer_size;j++){
        auto& temp_tensor = this->infer_status.status_recent.at(j).maskmem_pos_enc;
        auto sub = temp_tensor[0].GetTensorData<float>();//[4096,1,64]
        tensor_add(maskmem_pos_enc_+j*4096*64,sub,temporal_code.at(j));
    }
    // 第二部分：
    for(size_t i = maskmem_buffer_size*4096*64;i<maskmem_pos_enc_size;i++){
        maskmem_pos_enc_[i] = 0.0f;
    }
    // [z,1,64]
    std::vector<int64_t> dimensions_3{int64_t(maskmem_buffer_size*4096+4*std::min(infer_status.current_frame,16)),1,64};
    auto memory_pos_embed = Ort::Value::CreateTensor<float>(
                        memory_info,
                        maskmem_pos_enc_,
                        maskmem_pos_enc_size,
                        dimensions_3.data(),
                        dimensions_3.size()
                        );
    input_tensor.push_back(std::move(memory_pos_embed));
    return input_tensor;
}
// input [1,3,1024,1024]
// output: 
//      pix_feat        [1,256,64,64]
//      high_res_feat0  [1,32,256,256]
//      high_res_feat1  [1,64,128,128]
//      vision_feats    [1,256,64,64]
//      vision_pos_embed [4096,1,256]
std::variant<bool,std::string> SAM2::img_encoder_infer(std::vector<Ort::Value> &input_tensor){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->img_encoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->img_encoder_output_nodes) output_names.push_back(node.name);
    try {
        this->img_encoder_out = this->img_encoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(), 
            output_names.data(), 
            output_names.size()); 
    }catch (const std::exception& e) {
        std::string error(e.what());
        return std::format("ERROR: img_encoder_infer failed!!\n {}",error);
    }
    return true;
}

// input:
//      num_obj_ptr             [num]
//      current_vision_feat     [4096,1,256]
//      current_vision_pos_embed [4096,1,256]
//      memory                  [buff,1,64]
//      memory_pos_embed        [buff,1,64]
// output:
//      image_embed     [1,256,64,64]
std::variant<bool,std::string> SAM2::mem_attention_infer(){
    if(infer_status.current_frame == 0) [[unlikely]]{
        this->mem_attention_out.push_back(std::move(this->img_encoder_out[3]));
        return true;
    }
    //创建输入数据
    auto input_tensor = this->build_mem_attention_input();
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->mem_attention_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->mem_attention_output_nodes) output_names.push_back(node.name);
    try {
        this->mem_attention_out = this->mem_attention_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size());
    }catch (const std::exception& e) {
        std::string error(e.what());
        return std::format("ERROR: img_encoder_infer failed!!\n {}",error);
    }
    return true;
}

// input:
//      point_coords    [num_labels,num_points,2]
//      point_labels    [num_labels,num_points]
//      frame_size      [2]
//      image_embed     [1,256,64,64]
//      high_res_feats_0[1,32,256,256]
//      high_res_feats_1[1,64,128,128]
// output:
//      obj_ptr       [1,256]
//      mask_for_mem  [1,1,1024,1024]
//      pred_mask     [1,H,W]
std::variant<bool,std::string> SAM2::img_decoder_infer(){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->img_decoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->img_decoder_output_nodes) output_names.push_back(node.name);
    std::vector<Ort::Value> input_tensor; // 6
    auto& box = parms.prompt_box;
    std::vector<float>point_val{(float)box.x,(float)box.y,(float)box.x+box.width,(float)box.y+box.height};//xyxy
    std::vector<float> point_labels = {2,3};
    std::vector<int64> frame_size = {ori_img->rows,ori_img->cols};
    //***************************************************************
    this->img_decoder_input_nodes[0].dim = {1,2,2};
    this->img_decoder_input_nodes[1].dim = {1,2};
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info,point_val.data(),point_val.size(),
                        this->img_decoder_input_nodes[0].dim.data(),
                        this->img_decoder_input_nodes[0].dim.size()));
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info,point_labels.data(),point_labels.size(),
                        this->img_decoder_input_nodes[1].dim.data(),
                        this->img_decoder_input_nodes[1].dim.size()));
    input_tensor.push_back(Ort::Value::CreateTensor<int64>(memory_info,frame_size.data(),frame_size.size(),
                        this->img_decoder_input_nodes[2].dim.data(),
                        this->img_decoder_input_nodes[2].dim.size()));
    
    input_tensor.push_back(std::move(this->mem_attention_out[0]));
    input_tensor.push_back(std::move(this->img_encoder_out[1]));
    input_tensor.push_back(std::move(this->img_encoder_out[2]));
    //***************************************************
    try {
        this->img_decoder_out = std::move(this->img_decoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()));
    }catch (const std::exception& e) {
        std::string error(e.what());
        return std::format("ERROR: img_encoder_infer failed!!\n {}",error);
    }
    if(infer_status.current_frame == 0){
        infer_status.obj_ptr_first.push_back(std::move(this->img_decoder_out[0]));
    }else{
        infer_status.obj_ptr_recent.push(std::move(this->img_decoder_out[0]));
    }
    return true;
}

// input:
//      mask_for_mem    [1,1,1024,1024]
//      pix_feat        [1,256,64,64]
// output:
//      maskmem_features  [1,64,64,64]
//      maskmem_pos_enc   [4096,1,64]
//      temporal_code     [7,1,1,64]
std::variant<bool,std::string> SAM2::mem_encoder_infer(){
    std::vector<Ort::Value> input_tensor; // 2
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->mem_encoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->mem_encoder_output_nodes) output_names.push_back(node.name);
    input_tensor.push_back(std::move(this->img_decoder_out[1]));
    input_tensor.push_back(std::move(this->img_encoder_out[0]));
    //***************************************************
    try {
        this->mem_encoder_out = this->mem_encoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size());
    }catch (const std::exception& e) {
        std::string error(e.what());
        return std::format("ERROR: img_encoder_infer failed!!\n {}",error);
    }
    // 存储推理状态
    SubStatus temp;
    temp.maskmem_features.push_back(std::move(this->mem_encoder_out[0]));
    temp.maskmem_pos_enc.push_back(std::move(this->mem_encoder_out[1]));
    infer_status.status_recent.push(std::move(temp));
    return true;
}

void SAM2::preprocess(cv::Mat &image){
    input_images.clear();
    std::vector<cv::Mat> mats{image};
    cv::Mat img = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(1024,1024), cv::Scalar(0, 0, 0), true, false);
    input_images.push_back(std::move(img));
}

void SAM2::postprocess(std::vector<Ort::Value> &output_tensors){
    std::println("postprocess");
    float* output =  output_tensors[0].GetTensorMutableData<float>();
    cv::Mat outimg(this->ori_img->size(),CV_32FC1,output);
    cv::Mat dst;
    outimg.convertTo(dst, CV_8UC1, 255);
    cv::threshold(dst,dst,0,255,cv::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(*ori_img, contours, -1, cv::Scalar(50,250,20),1,cv::LINE_AA);
    // cv::imshow("*ori_img",*ori_img);
    // cv::waitKey(0);
}
int SAM2::setparms(ParamsSam2 parms){
    this->parms = std::move(parms);
    return 1;
}