from image import *
import os

def basic_test():
    # 原始图片
    # 加载图片的路径和名称
    img_path = 'A.png'

    # 读取原始图片
    img = read_image(img_path)

    # 展示原始图片
    plot_image(image=img, image_title="original image")

    # 生成受损图片
    # 图像数据归一化
    nor_img = normalization(img)

    # 噪声比率
    noise_ratio = 0.5

    # 生成受损图片
    noise_img = noise_mask_image(nor_img, noise_ratio)

    if noise_img is not None:
        # 展示受损图片
        plot_image(image=noise_img, image_title="the noise_ratio = %s of original image"%noise_ratio)

        # 恢复图片
        res_img = restore_image(noise_img)
        
        # 计算恢复图片与原始图片的误差
        print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
        print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
        print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))

        # 展示恢复图片
        plot_image(image=res_img, image_title="restore image")

        # 保存恢复图片
        save_image('res_' + img_path, res_img)
    else:
        # 未生成受损图片
        print("返回值是 None, 请生成受损图片并返回!")

# do A's variable noise test

def test_A():
    ratios = [0.4, 0.6, 0.8]
    img_path = './A.png'
    img = read_image(img_path)
    nor_img = normalization(img)

    # create path
    os.system('mkdir -p result/A')

    for noise_ratio in ratios:
        print("testing A of noise_ratio %s!" % noise_ratio)
        noise_img = noise_mask_image(nor_img, noise_ratio)
        if noise_img is not None:
            res_img = restore_image(noise_img)
            # 计算恢复图片与原始图片的误差
            print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
            print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
            print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))
            # 保存噪声图片
            save_image('./result/A/A_%s_noise.png'%noise_ratio, noise_img)
            # 保存恢复图片
            save_image('./result/A/A_%s_restore.png'%noise_ratio, res_img)
        else:
            # 未生成受损图片
            print("返回值是 None, 请生成受损图片并返回!")
        print("done!")
    print("test done!")

# more test
# save mid-stage images

def test_all():
    todos=['forest']
    # todos=['forest','mona_lisa','potala_palace','the_school_of_athens','xihu']
    for todo in todos:
        print("restoring %s image" % todo)
        # create path
        os.system('mkdir -p result/%s' % todo)
        origin_img_path = './samples/%s.png' % todo
        origin_img = read_image(origin_img_path)
        save_image('./result/%s/%s.png' % (todo, todo), origin_img)
        nor_img = normalization(origin_img)
        
        # noise img
        noise_img_path = './samples/%s_noise.png' % todo
        noise_img = read_image(noise_img_path)
        res_img = restore_image(noise_img)
        # 计算恢复图片与原始图片的误差
        print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
        print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
        print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))
        # 保存噪声图片
        save_image('./result/%s/%s_noise.png' % (todo, todo), noise_img)
        # 保存恢复图片
        save_image('./result/%s/%s_noise_restore.png' % (todo, todo), res_img)

        # noise random img
        random_noise_img_path = './samples/%s_random_noise.png' % todo
        random_noise_img = read_image(random_noise_img_path)
        res_img = restore_image(random_noise_img)
        # 计算恢复图片与原始图片的误差
        print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
        print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
        print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))
        # 保存噪声图片
        save_image('./result/%s/%s_random_noise.png' % (todo, todo), random_noise_img)
        # 保存恢复图片
        save_image('./result/%s/%s_random_noise_restore.png' % (todo, todo), res_img)

        print("done!")
    print("restoring done!")
    

if __name__ == "__main__":
    # basic_test()
    test_A()
    # test_all()




