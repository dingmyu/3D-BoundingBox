import os
import glob
import random
img_list = glob.glob('training/image_2/*.png')
img_list.sort()
img_dir = '/mnt/lustre/dingmingyu/Research/3dbbox/3D-Bbox/Kitti/'
for item in img_list:
	print img_dir + item.replace('image','prev').replace('.png','_01.png'), img_dir + item
	
