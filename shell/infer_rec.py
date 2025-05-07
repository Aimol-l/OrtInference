import cv2
import numpy as np
import onnxruntime



def inference(img_path, onnx_file):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用CUDA，如果不可用则回退到CPU
    session = onnxruntime.InferenceSession(onnx_file, providers=providers)
    img = cv2.imread(img_path)
    if img is None:  # 如果图片为空，则退出程序
        print("Error: Image not found.")
        return
    

    input_h, input_w = 48,320
    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    input_name = session.get_inputs()[0].name
    img = np.transpose(img, (2, 0, 1))# 转换维度顺序为CHW
    input_data = np.expand_dims(img, axis=0)
    input_data = input_data.astype(np.float32) / 255.0
    input_data -=0.5
    input_data /= 0.5

    pred = session.run(None, {input_name: input_data})[0] #(1, 1, 352, 480)

    print(pred.shape)
   

if __name__ == '__main__':

    img_path = './assets/ocr/2.png'
    onnx_file = './models/ocr/server/rec_zh.onnx'

    inference(img_path, onnx_file)