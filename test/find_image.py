import os


def get_image_path(image_name, image_dir):
    pathdir = os.listdir(image_dir)
    for p in pathdir:
        img_dir = os.path.join(image_dir, p)
        img_path = os.path.join(img_dir, image_name)
        if os.path.exists(img_path):
            return img_path


if __name__ == '__main__':
    file_name = '0019-1_1-340&500_404&526-404&524_340&526_340&502_404&500-0_0_11_26_25_28_17-66-3.jpg'
    base_path = 'F:\code_download\CCPD2019'

    path = []

    path = get_image_path(file_name, base_path)
    print(path)
