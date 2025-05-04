import cv2
import numpy as np
import onnxruntime



def inference(img_path, onnx_file):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先使用CUDA，如果不可用则回退到CPU
    session = onnxruntime.InferenceSession(onnx_file, providers=providers)
    img = cv2.imread(img_path)
    src_img = img.copy()
    if img is None:  # 如果图片为空，则退出程序
        print("Error: Image not found.")
        return
    input_name = session.get_inputs()[0].name
    # 将图片resize到32的倍数
    neww_h, new_w = 48,192#img.shape[1] // 32 * 32
    img = cv2.resize(img, (new_w, neww_h))
    img = np.transpose(img, (2, 0, 1))# 转换维度顺序为CHW
    input_data = np.expand_dims(img, axis=0)
    input_data = input_data.astype(np.float32) / 255.0

    output = session.run(None, {input_name: input_data})[0] #(1,2)
    print(output)
    cv2.imshow('img', src_img)
    cv2.waitKey(0)

if __name__ == '__main__':

    img_path = './assets/ocr/3.png'
    onnx_file = './models/ocr/mobile/cls.onnx'

    inference(img_path, onnx_file)