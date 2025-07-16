import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from PIL import Image
import random
import math
import numpy as np
# 定义命令行参数
parser = argparse.ArgumentParser(description='Convert image to grayscale')
parser.add_argument('--input', type=str, default='BBB1.jpg', help='输入图像路径')
parser.add_argument('--output', type=str, default='grayscale_image.jpg', help='输出图像路径')
parser.add_argument('--img_h', type=int, default=344, help='图像高度')
parser.add_argument('--img_w', type=int, default=127, help='图像宽度')
parser.add_argument('--erasing_p', type=float, default=0.5, help='随机擦除概率')
args = parser.parse_args()

# 定义颜色增强(这里简化处理)
color_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

# 定义归一化转换
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class PILRandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size[0] * img.size[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[0] and h < img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                # 转换为RGB模式处理
                if img.mode == 'L':
                    img_np = np.array(img)
                    img_np[y1:y1 + h, x1:x1 + w] = int(self.mean[0] * 255)
                    img = Image.fromarray(img_np)
                else:
                    img_np = np.array(img)
                    img_np[y1:y1 + h, x1:x1 + w, 0] = int(self.mean[0] * 255)
                    img_np[y1:y1 + h, x1:x1 + w, 1] = int(self.mean[1] * 255)
                    img_np[y1:y1 + h, x1:x1 + w, 2] = int(self.mean[2] * 255)
                    img = Image.fromarray(img_np)

                return img

        return img
# 定义图像转换
transform_sysu = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    color_aug,
    # transforms.Pad(10),
    transforms.RandomGrayscale(p=0.1),  # 设置p=1确保总是转换为灰度图
    transforms.RandomCrop((args.img_h, args.img_w)),
    # transforms.RandomHorizontalFlip(),
    # 使用自定义的PIL兼容RandomErasing
    # PILRandomErasing(probability=args.erasing_p, sl=0.2, sh=0.8, r1=0.3, mean=[0.485, 0.456, 0.406]),
    transforms.ToTensor(),
    normalize,
])


def convert_to_grayscale(input_path, output_path):
    # 读取图像
    image = Image.open(input_path).convert('RGB')

    # 应用转换
    transformed_image = transform_sysu(image)

    # 转换回PIL图像以保存
    # 移除归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    transformed_image = inv_normalize(transformed_image)

    # 将Tensor转换回PIL图像
    pil_image = transforms.ToPILImage()(transformed_image)

    # 保存图像
    pil_image.save(output_path)
    print(f"灰度图像已保存至: {output_path}")


if __name__ == "__main__":
    convert_to_grayscale(args.input, args.output)