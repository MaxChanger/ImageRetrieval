'''
data generator for the global local net

v0: for resnet34 only
v1: for global local with only local path, prepare the data for the input['local']
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import pandas as pd
import random
import os
import math
# from skimage import io, transform
import numpy as np
import cv2
from time import time
from PIL import Image
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
plt.ion()



class dataconfig(object):
    def __init__(self, dataset = 'defaut',subset = '0', **kwargs):
        self.dataset = dataset
        self.dir = r'E:\Code_develop\Rimag\ImageRetrieval-1\IMRA_model\dataset'
        self.csv = 'train_val.csv'
        self.subset = subset
        self.csv_file = os.path.join(self.dir,self.csv)

class batch_sampler():
    def __init__(self, batch_size, class_list):
        self.batch_size = batch_size
        self.class_list = class_list
        self.unique_value = np.unique(class_list)
        self.iter_list = []
        self.len_list = []
        for v in self.unique_value:
            indexes = np.where(self.class_list == v)[0]
            self.iter_list.append(self.shuffle_iterator(indexes))
            self.len_list.append(len(indexes))
        self.len = len(class_list) // batch_size
        # print('self.len: ', self.len)

    def __iter__(self):
        index_list = []
        for _ in range(self.len):
            for index in range(self.batch_size):
                index_list.append(next(self.iter_list[index % len(self.unique_value)]))
            np.random.shuffle(index_list)
            yield index_list
            index_list = []

    def __len__(self):
        return self.len

    @staticmethod
    def shuffle_iterator(iterator):
        # iterator should have limited size
        index = list(iterator)
        total_size = len(index)
        i = 0
        random.shuffle(index)
        while True:
            yield index[i]
            i += 1
            if i >= total_size:
                i = 0
                random.shuffle(index)


class DataGenerator(Dataset):
    def __init__(self, config=None,transform = None):
        self.config = config
        self.debug = False
        self.df = self.parse_csv(self.config.csv_file, self.config.subset)
        self.df.reset_index(drop=True, inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def img_augmentation(self, img, seq_det):

        img = img.transpose(2, 0, 1)

        for i in range(len(img)):
            img[i, :, :] = seq_det.augment_image(img[i, :, :])

        img = img.transpose(1, 2, 0)

        # img = seq_det.augment_images(img)

        return img

    def __getitem__(self, index):

        img_path = self.df.loc[index, 'data_path']
        # print(index,img_path)
        image = cv2.imread(img_path)
        # image = Image.open(img_path)
        label = self.df.loc[index, 'tag']
        # label = self.df.loc[index, 'Margin']
        #
        # label = label.reshape(-1,1)
        # landmarks = landmarks.reshape(-1, 2)
        # sample = {'image': image, 'label': label}

        if self.transform:

            image = np.array(image)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
            # image = np.stack([image, image, image], axis=0)
            # dec = random.choice(range(2))
            # if dec == 1 and self.df.loc[index, 'valid'] == 0:
            if self.df.loc[index, 'valid'] == 0:
                # print('{} is img_auged'.format(index))
                seq = iaa.SomeOf((4, 7), [
                    iaa.Fliplr(0.8),
                    iaa.Flipud(0.8),
                    # iaa.Multiply((0.8, 1.2)),
                    iaa.GaussianBlur(sigma=(0.0, 0.2)),
                    iaa.PiecewiseAffine((0.01, 0.2)),
                    iaa.Affine(
                        rotate=(-60, 60),
                        shear=(-10, 10),
                        scale=({'x': (0.8, 1.2), 'y': (0.8, 1.2)})  # to strentch the image along x,y axis
                    ),
                    iaa.WithChannels(
                        channels=[0],
                        children=iaa.SomeOf((3, 6), [
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 0.3)),  # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 7)),
                                # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 11)),
                                # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2)),  # sharpen images
                            iaa.Emboss(alpha=(0, 0.5), strength=(0.8, 1.2)),  # emboss images
                            # search either for all edges or for directed edges,
                            # blend the result with the original image using a blobby mask
                            # iaa.SimplexNoiseAlpha(iaa.OneOf([
                            #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                            # ])),
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255)),
                            # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.03)),  # randomly remove up to 10% of the pixels
                                # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05)),
                            ]),
                            iaa.Invert(0.05),  # invert color channels
                            iaa.Add((-10, 10)),
                            # change brightness of images (by -10 to 10 of original value)
                            # iaa.AddToHueAndSaturation((-20, 20))  # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                        ])
                    )

                ])

                seq_det = seq.to_deterministic()

                if len(np.shape(image)) == 2:
                    image = seq_det.augment_image(image)
                else:
                    image = seq_det.augment_image(image[:,:,0])
                # elif np.shape(image)[2] == 3:
                #     print(index, np.shape(image))
                #     image = self.img_augmentation(image, seq_det=seq_det)
                # else:
                #     print(index,np.shape(image))
                #     if np.shape(image)[1] == np.shape(image)[2]:
                #         image = seq_det.augment_image(image[0,:,:])
                #     else:
                #         image = seq_det.augment_image(image[:, :,0])


                # image = [image, image, image]

                # plt.imshow(image),plt.show()

        if self.transform:
            try:
                image = self.transform(image)
                if image.shape[0]==1:
                    image = torch.cat((image,image,image))
            except:
                print('something error:',index,img_path)
                plt.imshow(image), plt.show()

        if self.debug:
            print('data generator debug:',image.shape)

        return image,label

    @staticmethod
    def parse_csv(csv_file, subset):
        data_frame = pd.read_csv(csv_file)
        data_frame = data_frame[data_frame['valid'] == int(subset)]
        return data_frame


def show_landmarks(image, landmarks):
    """SHow image with landmarks"""
    plt.imshow(image)
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")


if __name__ == "__main__":
    # config = {"aug": True, "subset": 'training', "save_img": True, "add_noise": False}
    # config = {"dataset": 'mammo_calc',"subset": '0'}
    # train_config = dataconfig(**config)
    # train_dataset = DataGenerator(train_config,transform= transforms.ToTensor())
    #
    #
    # train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=4,shuffle= True)
    #
    # num_classes = 1
    # model = resnet2d.ResNet(dataset='calc', depth=34, num_classes=num_classes)
    #
    # criterion = torch.nn.BCELoss().cuda()
    #
    # # print(train_dataloader.batch_size)
    #
    # for i, (images,labels) in enumerate(train_dataloader):
    #     # print(sample['image'])
    #     outputs = model(images)
    #     labels = labels.float().reshape(-1,1)
    #     print(outputs.shape,labels.shape)
    #     loss = criterion(outputs,labels)
    #     print('loss: ',loss)

    valconfig = {"dataset": "calc","subset": '1'}
    val_config = dataconfig(**valconfig)
    validation_data = DataGenerator(val_config,transform= transforms.ToTensor())
    val_loader = DataLoader(validation_data,batch_size=32, num_workers=1,shuffle=True)

    for i, (images, labels) in enumerate(val_loader):
        print(i)
        print(labels)
        print(images.shape)

