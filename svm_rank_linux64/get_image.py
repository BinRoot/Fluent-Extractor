import numpy as np
from matplotlib import pyplot as plt

# given meta information about the image {vid#, img#}
# return the image
def load_image(img_meta):  # 1:45
	# TODO: make this a relative path
	base_path = '/home/kfrankc/Desktop/cloth_folding_lite/'
	img_meta_parts = img_meta.split(':')
	vid_id, img_id = img_meta_parts[0], img_meta_parts[1]
	img_path = base_path + vid_id + '/' + 'aligned_rgb_' + '0'.zfill(5-len(img_id)) + img_id + '.png'
	print img_path
	image_ret = plt.imread(img_path)
	return image_ret