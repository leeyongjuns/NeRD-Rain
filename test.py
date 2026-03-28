import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from model import MultiscaleNet as mynet
from skimage import img_as_ubyte
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Image Deraining and Person Detection')
    parser.add_argument('--input_dir', default='Datasets/Rain200L/test/input', type=str,
                        help='输入图像文件夹路径')
    parser.add_argument('--output_dir', default='result', type=str,
                        help='输出文件夹路径')
    parser.add_argument('--derain_weights', default='model/train/models/rain200L/model_best.pth', type=str,
                        help='去雨模型权重路径')
    parser.add_argument('--yolo_weights', default='yolov8s-world.pt', type=str,
                        help='YOLO-World模型权重路径')
    parser.add_argument('--gpus', default='0', type=str,
                        help='使用的GPU设备')
    parser.add_argument('--win_size', default=256, type=int,
                        help='去雨模型窗口大小')
    args = parser.parse_args()

    # 设置设备
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True

    # 初始化去雨模型
    model_restoration = mynet()
    get_parameter_number(model_restoration)
    utils.load_checkpoint(model_restoration, args.derain_weights)
    print("===> 使用去雨模型权重测试: ", args.derain_weights)

    model_restoration = nn.DataParallel(model_restoration).cuda()
    model_restoration.eval()

    # 初始化YOLO-World模型
    model_yolo = YOLOWorld(args.yolo_weights)
    model_yolo.set_classes(["person"])

    # 准备数据集
    test_dataset = get_test_data(args.input_dir, img_options={})
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 适配Windows环境
        drop_last=False,
        pin_memory=True
    )

    # 创建输出目录
    utils.mkdir(args.output_dir)
    person_dir = os.path.join(args.output_dir, 'person')
    utils.mkdir(person_dir)

    # 测试循环
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]
            _, _, Hx, Wx = input_.shape

            input_re, batch_list = window_partitionx(input_, args.win_size)
            restored = model_restoration(input_re)
            restored = window_reversex(restored[0], args.win_size, Hx, Wx, batch_list)

            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                # 临时保存去雨后的图像
                temp_img_path = os.path.join(args.output_dir, 'temp.png')
                utils.save_img(temp_img_path, restored_img)

                # 加载去雨后的图像用于YOLO-World检测
                img = Image.open(temp_img_path)

                # 进行人体检测
                results = model_yolo.predict(img)

                # 获取检测结果的图像（NumPy 数组）
                result_img = results[0].plot()

                # 将 NumPy 数组转换为 PIL 图像对象
                if result_img.dtype != np.uint8:
                    result_img = (result_img * 255).astype(np.uint8)  # 确保数据类型为 uint8
                result_img_pil = Image.fromarray(result_img)

                # 保存检测结果
                result_img_pil.save(os.path.join(person_dir, filenames[batch] + '_person.png'))

                # 删除临时图像
                os.remove(temp_img_path)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # 适配Windows环境
    main()