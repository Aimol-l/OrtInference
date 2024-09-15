#include "SAM2.h"

std::variant<bool,std::string> SAM2::initialize(std::vector<std::string>& onnx_paths, bool is_cuda){
    // image_encoder,memory_attention,image_decoder,memory_encoder
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

            // const auto& api = Ort::GetApi();
            // OrtTensorRTProviderOptionsV2* tensorrt_options;
            // Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
            // std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
            // Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(this->img_encoder_options),rel_trt_options.get()));
            // Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(this->img_decoder_options),rel_trt_options.get()));
            // Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(this->mem_attention_options),rel_trt_options.get()));
            // Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(this->mem_encoder_options),rel_trt_options.get()));
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
        unsigned len_img_encoder = onnx_paths[0].size() * 2; // 预留字节数
        unsigned len_mem_attention = onnx_paths[1].size() * 2; // 预留字节数
        unsigned len_img_decoder = onnx_paths[2].size() * 2; // 预留字节数
        unsigned len_mem_encoder = onnx_paths[3].size() * 2; // 预留字节数
        setlocale(LC_CTYPE, ""); //必须调用此函数,本地化
        wchar_t* p_img_encoder = new wchar_t[len_yolo]; // 申请一段内存存放转换后的字符串
        wchar_t* p_mem_attention = new wchar_t[len_encoder]; // 申请一段内存存放转换后的字符串
        wchar_t* p_img_decoder = new wchar_t[len_decoder]; // 申请一段内存存放转换后的字符串
        wchar_t* p_mem_encoder = new wchar_t[len_decoder]; // 申请一段内存存放转换后的字符串

        mbstowcs(p_img_encoder, onnx_paths[0].c_str(), len_img_encoder); // 转换
        mbstowcs(p_mem_attention,onnx_paths[1].c_str(), len_mem_attention); // 转换
        mbstowcs(p_img_decoder, onnx_paths[2].c_str(), len_img_decoder); // 转换
        mbstowcs(p_mem_encoder, onnx_paths[3].c_str(), len_mem_encoder); // 转换

        std::wstring wstr_img_encoder(p_img_encoder);
        std::wstring wstr_mem_attention(p_mem_attention);
        std::wstring wstr_img_decoder(p_img_decoder);
        std::wstring wstr_mem_encoder(p_mem_encoder);

        delete[] p_img_encoder,p_mem_attention,p_img_decoder,p_mem_encoder; // 释放申请的内存

        img_encoder_session = new Ort::Session(img_encoder_env, wstr_img_encoder.c_str(), this->img_encoder_options);
        mem_attention_session = new Ort::Session(mem_attention_env, wstr_mem_attention.c_str(), this->mem_attention_options);
        img_decoder_session = new Ort::Session(img_decoder_env, wstr_img_decoder.c_str(), this->img_decoder_options);
        mem_encoder_session = new Ort::Session(mem_encoder_env, wstr_mem_encoder.c_str(), this->mem_encoder_options);
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
    // this->input_images.resize(3*1024*1024);
    std::println("initialize ok!!");
    return true;
}

std::variant<bool,std::string> SAM2::inference(cv::Mat &image){
    if (image.empty() || !is_inited) return "image can not empyt!";
    this->ori_img = &image;
    // 图片预处理
    try {
        this->preprocess(image); // 
     }catch (const std::exception& e) {
        return "Image preprocess failed!";
    }
    // 图片编码器，输入图片
    std::vector<Ort::Value> img_encoder_tensor;
    img_encoder_tensor.push_back(std::move(Ort::Value::CreateTensor<float>(
                        memory_info,
                        this->input_images[0].ptr<float>(),
                        this->input_images[0].total(),
                        this->img_encoder_input_nodes[0].dim.data(),
                        this->img_encoder_input_nodes[0].dim.size()))
    );
    //*****************************img_encoder**********************************
    auto result_0 = this->img_encoder_infer(img_encoder_tensor);
    if(result_0.index() != 0) return std::get<std::string>(result_0);
    auto& img_encoder_out =  std::get<0>(result_0); // pix_feat,high_res_feat0,high_res_feat1,vision_feats,vision_pos_embed
    
    //******************************mem_attention*********************************
    auto result_1 = this->mem_attention_infer(img_encoder_out); 
    if(result_1.index() != 0) return std::get<std::string>(result_1);
    auto& mem_attention_out = std::get<0>(result_1); // image_embed
    
    //*****************************img_decoder**********************************
    mem_attention_out.push_back(std::move(img_encoder_out[1])); // high_res_feat0
    mem_attention_out.push_back(std::move(img_encoder_out[2])); // high_res_feat1
    auto result_2 =this->img_decoder_infer(mem_attention_out);
    if(result_2.index() != 0) return std::get<std::string>(result_2);
    auto& img_decoder_out = std::get<0>(result_2); // obj_ptr,mask_for_mem,pred_mask
    
    //***********************************************************************
    if(infer_status.current_frame == 0)[[unlikely]]{
        infer_status.obj_ptr_first.push_back(std::move(img_decoder_out[0]));
    }else{
        infer_status.obj_ptr_recent.push(std::move(img_decoder_out[0]));
    }
    //*******************************mem_encoder******************************
    std::vector<Ort::Value> mem_encoder_in;
    mem_encoder_in.push_back(std::move(img_decoder_out[1])); //mask_for_mem
    mem_encoder_in.push_back(std::move(img_encoder_out[0])); //pix_feat
    auto result_3 =this->mem_encoder_infer(mem_encoder_in);
    if(result_3.index() != 0) return std::get<std::string>(result_3);
    auto& mem_encoder_out = std::get<0>(result_3); // maskmem_features,maskmem_pos_enc,temporal_code
    //***************************************************************
    // 存储推理状态
    SubStatus temp;
    temp.maskmem_features.push_back(std::move(mem_encoder_out[0]));
    temp.maskmem_pos_enc.push_back(std::move(mem_encoder_out[1]));
    temp.temporal_code.push_back(std::move(mem_encoder_out[2]));
    infer_status.status_recent.push(std::move(temp));
    //***************************************************************
    // 后处理
    std::vector<Ort::Value> output_tensors;
    output_tensors.push_back(std::move(img_decoder_out[2])); //pred_mask
    try {
        this->postprocess(output_tensors);
    }catch (const std::exception& e) {
        return "tensor postprocess failed!!";
    }
    this->infer_status.current_frame++;
    return true;
}
// input [1,3,1024,1024]
// output: 
//      pix_feat        [1,256,64,64]
//      high_res_feat0  [1,32,256,256]
//      high_res_feat1  [1,64,128,128]
//      vision_feats    [1,256,64,64]
//      vision_pos_embed [4096,1,256]
std::variant<std::vector<Ort::Value>,std::string> SAM2::img_encoder_infer(std::vector<Ort::Value> &input_tensor){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->img_encoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->img_encoder_output_nodes) output_names.push_back(node.name);

    std::vector<Ort::Value> img_encoder_out;
    try {
        img_encoder_out = std::move(this->img_encoder_session->Run(
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
    return img_encoder_out;
}

// input: 
//    pix_feat          [1,256,64,64]
//    high_res_feat0    [1,32,256,256]
//    high_res_feat1    [1,64,128,128]
//    vision_feats      [1,256,64,64]
//    vision_pos_embed  [4096, 1, 256]
// out:
//    image_embed   [1,256,64,64]
std::variant<std::vector<Ort::Value>,std::string> SAM2::mem_attention_infer(std::vector<Ort::Value>&img_encoder_out){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->mem_attention_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->mem_attention_output_nodes) output_names.push_back(node.name);
    std::vector<Ort::Value> mem_attention_out;
    if(infer_status.current_frame == 0) [[unlikely]]{
        mem_attention_out.push_back(std::move(img_encoder_out[3]));
        return mem_attention_out;
    }
    //*******************************************************************************
    //创建输入数据 current_vision_feat，current_vision_pos_embed，memory_0，memory_1，memory_pos_embed
    std::vector<Ort::Value> input_tensor; // 5
    input_tensor.push_back(std::move(img_encoder_out[3])); //current_vision_feat
    input_tensor.push_back(std::move(img_encoder_out[4])); //current_vision_pos_embed
    // 需要构造出 memory 和 memory_pos_embed
    // memory是由obj_ptr_first，obj_ptr_recent和status_recent.maskmem_features 构造出的
    size_t obj_buffer_size = 1 + infer_status.obj_ptr_recent.size();//1+0,1+1,1+2,...,1+15

    std::vector<int64_t> dimensions_0{(int64_t)obj_buffer_size,256}; // [y,256]
    std::vector<float> obj_ptrs(obj_buffer_size*256); // first+recent // 16*256

    const float* tensor_data = infer_status.obj_ptr_first[0].GetTensorData<float>();
    std::copy_n(tensor_data, 256, std::begin(obj_ptrs));

    for(size_t i = 0;i<infer_status.obj_ptr_recent.size();i++){
        auto& temp_tensor = infer_status.obj_ptr_recent.at(i);
        tensor_data = temp_tensor.GetTensorData<float>();
        std::copy_n(tensor_data, 256, std::begin(obj_ptrs)+256*(i+1));
    }

    auto memory_0 = Ort::Value::CreateTensor<float>(
                    memory_info,
                    obj_ptrs.data(),
                    obj_ptrs.size(),
                    dimensions_0.data(),
                    dimensions_0.size()
                    );
    
    size_t features_size = infer_status.status_recent.size(); // 1,2,3,...,7
    std::vector<float> maskmem_features_(features_size*64*64*64);
    for(size_t i = 0;i<features_size;i++){
        auto& temp_tensor = this->infer_status.status_recent.at(i).maskmem_features;
        tensor_data = temp_tensor[0].GetTensorData<float>();
        std::copy_n(tensor_data, 64*64*64, std::begin(maskmem_features_)+64*64*64*i);
    }
    std::vector<int64_t> dimensions_1{(int64_t)features_size,64,64,64}; // [x,64,64,64]
    auto memory_1 = Ort::Value::CreateTensor<float>(
                    memory_info,
                    maskmem_features_.data(),
                    maskmem_features_.size(),
                    dimensions_1.data(),
                    dimensions_1.size()
                    );
    input_tensor.push_back(std::move(memory_0));
    input_tensor.push_back(std::move(memory_1));

    //***********************************************************************
    // memory_pos_embed是由两部分组成的。
    auto& temp_time = infer_status.status_recent.at(features_size-1).temporal_code;
    const float* temporal_code_ = temp_time[0].GetTensorData<float>(); // [7,64]
    std::vector<const float*> temporal_code;
    for(int i = 6;i>=0;i--){
        auto temp = temporal_code_+i*64;
        temporal_code.push_back(temp);
    }
    size_t maskmem_buffer_size = infer_status.status_recent.size();
    size_t maskmem_pos_enc_size = (maskmem_buffer_size*4096+4*std::min(infer_status.current_frame,16))*64;
    
    std::vector<float> maskmem_pos_enc_(maskmem_pos_enc_size);
    
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
        float* p = maskmem_pos_enc_.data() + j*4096*64;
        tensor_add(p,sub,temporal_code.at(j)); // [4096,1,64] + [1,1,64] ->[4096,1,64] + [4096,1,64] ->[4096,1,64]
    }
    // 第二部分：
   std::fill_n(maskmem_pos_enc_.begin()+maskmem_buffer_size*4096*64,maskmem_pos_enc_size - (maskmem_buffer_size*4096*64), 0.0f);
    
    // [z,1,64]
    std::vector<int64_t> dimensions_3{int64_t(maskmem_buffer_size*4096+4*std::min(infer_status.current_frame,16)),1,64};
    auto memory_pos_embed = Ort::Value::CreateTensor<float>(
                        memory_info,
                        maskmem_pos_enc_.data(),
                        maskmem_pos_enc_.size(),
                        dimensions_3.data(),
                        dimensions_3.size()
                        );
    input_tensor.push_back(std::move(memory_pos_embed));
    //*******************************************************************************
    try {
        mem_attention_out = std::move(this->mem_attention_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()));
    }catch (const std::exception& e) {
        std::string error(e.what());
        return std::format("ERROR: mem_attention_infer failed!!\n {}",error);
    }
    return mem_attention_out;
}

// input:
//      image_embed         [1,256,64,64]
//      high_res_feats_0    [1,32,256,256]
//      high_res_feats_1    [1,64,128,128]
// output:
//      obj_ptr       [1,256]
//      mask_for_mem  [1,1,1024,1024]
//      pred_mask     [1,H,W]
std::variant<std::vector<Ort::Value>,std::string> SAM2::img_decoder_infer(std::vector<Ort::Value>&mem_attention_out){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->img_decoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->img_decoder_output_nodes) output_names.push_back(node.name);
    // point_coords,point_labels,frame_size,image_embed,high_res_feats_0,high_res_feats_1
    std::vector<Ort::Value> input_tensor; // 6
    auto box = parms.prompt_box;
    auto point = parms.prompt_point;
    // 变化bbox比例
    box.x = 1024*((float)box.x / ori_img->cols);
    box.y = 1024*((float)box.y / ori_img->rows);
    box.width = 1024*((float)box.width / ori_img->cols);
    box.height = 1024*((float)box.height / ori_img->rows);
    point.x = 1024*((float)point.x / ori_img->cols);
    point.y = 1024*((float)point.y / ori_img->rows);
    std::vector<float>point_val,point_labels;
    if(parms.type == 0){
        point_val = {(float)box.x,(float)box.y,(float)box.x+box.width,(float)box.y+box.height};//xyxy
        point_labels = {2,3};
        this->img_decoder_input_nodes[0].dim = {1,2,2};
        this->img_decoder_input_nodes[1].dim = {1,2};
    }else if(parms.type == 1){
        point_val = {(float)point.x,(float)point.y};//xy
        point_labels = {1};
        this->img_decoder_input_nodes[0].dim = {1,1,2};
        this->img_decoder_input_nodes[1].dim = {1,1};
    }
    std::vector<int64> frame_size = {ori_img->rows,ori_img->cols};
    //***************************************************************
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info,point_val.data(),point_val.size(),
                        this->img_decoder_input_nodes[0].dim.data(),
                        this->img_decoder_input_nodes[0].dim.size()));
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info,point_labels.data(),point_labels.size(),
                        this->img_decoder_input_nodes[1].dim.data(),
                        this->img_decoder_input_nodes[1].dim.size()));
    input_tensor.push_back(Ort::Value::CreateTensor<int64>(memory_info,frame_size.data(),frame_size.size(),
                        this->img_decoder_input_nodes[2].dim.data(),
                        this->img_decoder_input_nodes[2].dim.size()));
    
    input_tensor.push_back(std::move(mem_attention_out[0]));    // image_embed
    input_tensor.push_back(std::move(mem_attention_out[1]));    // high_res_feats_0
    input_tensor.push_back(std::move(mem_attention_out[2]));    // high_res_feats_1
    //***************************************************
    std::vector<Ort::Value> img_decoder_out;
    try {
        img_decoder_out = std::move(this->img_decoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()));
    }catch (const std::exception& e) {
        std::string error(e.what());
        return std::format("ERROR: img_decoder_infer failed!!\n {}",error);
    }
    return img_decoder_out;
}

// input:
//      mask_for_mem    [1,1,1024,1024]
//      pix_feat        [1,256,64,64]
// output:
//      maskmem_features  [1,64,64,64]
//      maskmem_pos_enc   [4096,1,64]
//      temporal_code     [7,1,1,64]
std::variant<std::vector<Ort::Value>,std::string> SAM2::mem_encoder_infer(std::vector<Ort::Value>&img_decoder_out){
    std::vector<const char*> input_names,output_names;
    for(auto &node:this->mem_encoder_input_nodes)  input_names.push_back(node.name);
    for(auto &node:this->mem_encoder_output_nodes) output_names.push_back(node.name);
    //***************************************************
    std::vector<Ort::Value> mem_encoder_out;
    try {
        mem_encoder_out = std::move(this->mem_encoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            img_decoder_out.data(),
            img_decoder_out.size(),
            output_names.data(),
            output_names.size()));
    }catch (const std::exception& e) {
        std::string error(e.what());
        return std::format("ERROR: mem_encoder_infer failed!!\n {}",error);
    }
    return mem_encoder_out;
}

void SAM2::preprocess(cv::Mat &image){
    std::vector<cv::Mat> mats{image};
    cv::Mat blob = cv::dnn::blobFromImages(mats, 1/255.0,cv::Size(1024,1024), cv::Scalar(0, 0, 0), true, false);
    input_images.clear();
    this->input_images.emplace_back(blob);
}

void SAM2::postprocess(std::vector<Ort::Value> &output_tensors){
    float* output =  output_tensors[0].GetTensorMutableData<float>();
    cv::Mat outimg(this->ori_img->size(),CV_32FC1,output);
    cv::Mat dst;
    outimg.convertTo(dst, CV_8UC1, 255);
    cv::threshold(dst,dst,0,255,cv::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);
    std::vector<std::vector<cv::Point>> contours; // 不一定是1
    cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int idx = -1;
    cv::Rect min_dis_rect;
    double min_dis = std::numeric_limits<double>::max();
    // 计算与 A 中心距离最近的 bbox
    for (size_t i = 0;i<contours.size();i++) {
        cv::Rect bbox = cv::boundingRect(contours[i]);
        cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        cv::Point A_center(parms.prompt_box.x + parms.prompt_box.width / 2, parms.prompt_box.y + parms.prompt_box.height / 2);
        double distance = cv::norm(center - A_center);
        if (distance < min_dis) {
            min_dis = distance;
            min_dis_rect = bbox;
            idx = i;
        }
    }
    if (!min_dis_rect.empty()) {
        parms.prompt_box = min_dis_rect;
        parms.prompt_point.x = min_dis_rect.x + min_dis_rect.width/2;
        parms.prompt_point.y = min_dis_rect.y + min_dis_rect.height/2;
        cv::drawContours(*ori_img, contours, idx, cv::Scalar(50,250,20),2,cv::LINE_AA);
        cv::rectangle(*ori_img, parms.prompt_box,cv::Scalar(0,0,255),2);
    }
}
int SAM2::setparms(ParamsSam2 parms){
    this->parms = std::move(parms);
    return 1;
}