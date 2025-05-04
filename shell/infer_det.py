import cv2
import numpy as np
import onnxruntime



def inference(img_path, onnx_file):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用CUDA，如果不可用则回退到CPU
    session = onnxruntime.InferenceSession(onnx_file, providers=providers)
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    src_img = img.copy()
    if img is None:  # 如果图片为空，则退出程序
        print("Error: Image not found.")
        return

    input_name = session.get_inputs()[0].name
    
    # 将图片resize到32的倍数

    neww_h, new_w = (img.shape[0] // 32) * 32, (img.shape[1] // 32) * 32
    img = cv2.resize(img, (new_w, neww_h))
    img = np.transpose(img, (2, 0, 1))# 转换维度顺序为CHW
    input_data = np.expand_dims(img, axis=0)
    input_data = input_data.astype(np.float32) / 255.0
    # input_data -=0.5
    # input_data /= 0.5

    output = session.run(None, {input_name: input_data})[0] #(1, 1, 352, 480)

    # (1, 1, 352, 480) --> (1, 352, 480) --> (352, 480)
    output = np.squeeze(output, axis=0)
    mask = np.transpose(output, (1,2,0))
    # mask = cv2.resize(mask, (src_w*2, src_h*2))
    # 显示img
    cv2.imshow('img', src_img)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

if __name__ == '__main__':

    img_path = './assets/ocr/1.jpg'
    onnx_file = './models/ocr/mobile/det.onnx'

    inference(img_path, onnx_file)