from PIL import Image
import numpy as np
from image import *

def read_img(path):
    img = Image.open(path)
    img = img.resize((150,150))
    img = np.asarray(img, dtype="uint8")
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(img.dtype)
    # 图像数组数据放缩在 0-1 之间
    return img.astype(np.double) / info.max


img =  read_img('A.png')
noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
noise[np.random.random(size=noise.shape) > 0.5] *= -1

img_noise = img + abs(noise)

print('相同图片的 SSIM 相似度: ', calc_ssim(img, img))
print('相同图片的 Cosine 相似度: ', calc_csim(img, img))
print('与噪声图片的 SSIM 相似度: ', calc_ssim(img, img_noise))
print('与噪声图片的 Cosine 相似度: ', calc_csim(img, img_noise))