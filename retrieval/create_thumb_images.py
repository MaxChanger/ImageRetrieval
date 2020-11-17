import os
import cv2


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def create_thumb_images(full_folder, thumb_folder, suffix='thumb', height=100, del_former_thumb=False):
    if not os.path.exists(thumb_folder):
        os.makedirs(thumb_folder)
    if del_former_thumb:
        del_file(thumb_folder)
    count = 0
    for image_file in os.listdir(full_folder):
        try:
            image = cv2.imread(os.path.join(full_folder, image_file))
            height_src, width_src, _ = image.shape
            #print('width: {}, height: {}'.format(width_src, height_src))

            width = (height / height_src) * width_src
            # print(' Thumb width: {}, height: {}'.format(width, height))

            resized_image = cv2.resize(image, (int(width), int(height)))

            image_name, image_extension = os.path.splitext(image_file)
            cv2.imwrite(os.path.join(thumb_folder, image_name + suffix + image_extension), resized_image)
        except Exception as e:
            count += 1
            os.remove(os.path.join(full_folder, image_file))
            print(count,os.path.join(full_folder, image_file), e)
            # print(e)
    print('Creating thumb images finished.')


if __name__ == '__main__':
    create_thumb_images(full_folder='./save_pic/',
                        thumb_folder='./thumb_images/',
                        suffix='',
                        height=200,
                        del_former_thumb=True,
                        )
