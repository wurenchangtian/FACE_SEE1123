import os
import numpy as np
import cv2

# 边距剪裁图像
IMAGE_SIZE = 64

def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    def get_padding_size(image):
        h, w, _ = image.shape           # 获取图像的原始长宽
        longest_edge = max(h, w)        # 获取最长边
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h       # 差值为最大边和高的差
            top = dh // 2
            bottom = dh - top

        elif w < longest_edge:
            dw = longest_edge - w       # 边缘剪裁
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right
    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    resize_image = cv2.resize(constant, (height, width))     # 重新定义图像大小
    return resize_image

# 定义图像数据读取的遍历函数，用于返回图像数据的标签并将图像数据读如内存
# 由于图像不是很多，直接一次性全部载入
images = []
labels = []

# 为每一类数据赋予唯一的标签值
def label_id(label,users,user_num):
    for i in range(user_num):
        if label.endswith(users[i]):
            return i


def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)                         # 打印绝对路径
        if os.path.isdir(abs_path):
            traverse_dir(abs_path)
        else:
            if file_or_dir.endswith('.jpg'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)
    return images, labels


def read_image(file_path):

    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    return image

# 定义数据标签提取函数


def extract_data(path):
    users = os.listdir(path)
    user_num = len(users)

    images, labels = traverse_dir(path)
    images = np.array(images)
    # labels = np.array([0 if label.endswith('AIGIRL') else 1 for label in labels])     # 简单是否分裂
    labels = np.array([label_id(label, users, user_num) for label in labels])             # 多分类，可以识别多个人
    return images, labels