#coding=utf-8
#Created by Administrator on 2017/8/10.

import os
import random
from PIL import Image, ImageFilter

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def blur_pics(sourcePath, targetPath):
	if not os.path.exists(targetPath):
		os.makedirs(targetPath)

	Index = 0
	for clear_pic in os.listdir(sourcePath):
		clear_img = os.path.join(sourcePath, clear_pic)
		blur_img = os.path.join(targetPath, clear_pic.split('.')[0]+'_blur.jpeg')

		img = Image.open(clear_img)
		img = img.filter(MyGaussianBlur(radius=random.randint(3,10)))
		img.save(blur_img)
		Index = Index+1
		print(Index)


if __name__=='__main__':
	srcPath = 'C:\\Users\Administrator\Desktop\data\BlurTest\\test_clear'#'C:\\Users\Administrator\Desktop\data\\blur_clear_pics\clear_pics'
	dstPath = 'C:\\Users\Administrator\Desktop\data\BlurTest\\test_blur'#'C:\\Users\Administrator\Desktop\data\\blur_clear_pics\\blur_pics'
	blur_pics(srcPath, dstPath)
	# simg = 'face_0.jpeg'
	# dimg = 'face_0_blur.jpeg'
	# image = Image.open(simg)
	# image = image.filter(MyGaussianBlur(radius=15))

# spainwang: 如果只需要处理某个区域，传入bounds参数即可
# bounds = (150, 130, 280, 230)
# image = image.filter(MyGaussianBlur(radius=29, bounds=bounds))
# image.show()

	# image.save(dimg)
	# print(dimg, 'success')