from image import *

img_path = 'samples/forest.png'

img = read_image(img_path)
plot_image(image=img, image_title="original image")
nor_img = normalization(img)

noise_img = read_image('samples/forest_noise.png')
plot_image(image=noise_img, image_title="noise image")
noise_img=normalization(noise_img)
# 恢复图片
res_img = restore_image(noise_img)

# 计算恢复图片与原始图片的误差
print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))

# 展示恢复图片
plot_image(image=res_img, image_title="restore image")

# 保存恢复图片
save_image('res_forest.png', res_img)