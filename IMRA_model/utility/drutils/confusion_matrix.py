import os
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class ConfusionMatrixGenerater(object):
    def __init__(self, image_data, csvfile):
        self.image_data = image_data
        self.csvfile = csvfile
        self.df = pd.read_csv(csvfile)
        self.biradsmap = {0.0: "bi-rads0", 1.0:"bi-rads1", 2.0:"bi-rads2", 3.0:"bi-rads3", 4.0:"bi-rads4",
                          4.1: "bi-rads4a", 4.2:"bi-rads4b", 4.3:"bi-rads4c", 5.0:"bi-rads5"}

    def gen_dict_from_csv(self):
        """
        create a dictionary that stores ziwei's birads label
        key: image name "pid-laterality", e.g. 111-L
        value: birads-num as csv file showed
        choose the highest birads to store
        """
        ziwei_birads = {}
        for i in range(len(self.df["index"])):
            patient_idx = self.df["index"][i].split("-")[0]
        #     laterality_idx = df["index"][i].split("-")[1]
            if self.df["index"][i].split("-")[1] == str(1) or self.df["index"][i].split("-")[1] == str(3):
                laterality = "R"
            elif self.df["index"][i].split("-")[1] == str(2) or self.df["index"][i].split("-")[1] == str(4):
                laterality = "L"
            else:
                print("wrong laterality ", self.df["index"][i])
            image_idx = patient_idx + "-" + laterality
            if image_idx not in ziwei_birads.keys():
                ziwei_birads[image_idx] = float(self.df["birads_num"][i])
            else:
                if ziwei_birads[image_idx] < float(self.df["birads_num"][i]):
                    ziwei_birads[image_idx] = float(self.df["birads_num"][i])
        """
        convert the num to real birads class, like bi-rads2, etc
        """
        ziwei_birads_new = {}
        for patient in ziwei_birads:
            ziwei_birads_new[patient] = self.biradsmap[ziwei_birads[patient]]
        return ziwei_birads_new

    def gen_dict_from_txt(self):
        """
        create a dictionary that stores the original comments.txt file info
        key: image name "pid-laterality", e.g. 111-L
        value: birads
        """
        image_comment = {}
        for patient in os.listdir(self.image_data):
            if os.path.isdir(os.path.join(self.image_data, patient)):
                comment_path = os.path.join(self.image_data, patient, "comments.txt")
                # print(comment_path)
                with open(comment_path, "r", encoding='utf-8') as f:
                    info = f.readlines()
                for i in range(len(info)):
                    if "left" in info[i].lower():
                        left_birads = info[i].split(":")[-1].lower().replace(" ", "").replace("\n","")
                        image_comment[patient+"-L"] = left_birads
                    if "right" in info[i].lower():
                        right_birads = info[i].split(":")[-1].lower().replace(" ", "").replace("\n","")
                        image_comment[patient+"-R"] = right_birads
        return image_comment

    def gen_confustion_matrix(self, class_names):
        csv_dict = self.gen_dict_from_csv()
        txt_dict = self.gen_dict_from_txt()
        ziwei_list = []
        img_comment_list = []
        # wrong_list = []
        count = 0
        for img in csv_dict.keys():
            if img in txt_dict.keys():
                count += 1
                ziwei_list.append(csv_dict[img])
                img_comment_list.append(txt_dict[img])
        cnf_matrix = confusion_matrix(img_comment_list, ziwei_list, labels=class_names)
        np.set_printoptions(precision=2)
        return cnf_matrix

    @staticmethod
    def plot_confusion_matrix(cm, classes, xlabel, ylabel,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        generate two lists that stores the bi-rads info
        corresponding info stores at the same slot index
        """
        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()

if __name__ == "__main__":
    img_data = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/ChinaLQ_Display_1"
    csvfile = r"/media/Data/Data02/Datasets/Mammogram/Ziwei_WIP/china_bbox.csv"
    class_names = ["bi-rads0", "bi-rads1", "bi-rads2", "bi-rads3", "bi-rads4", "bi-rads4a", "bi-rads4b",
                   "bi-rads4c", "bi-rads5"]
    cmg = ConfusionMatrixGenerater(img_data, csvfile)
    cnf_matrix = cmg.gen_confustion_matrix(class_names)
    cmg.plot_confusion_matrix(cnf_matrix, class_names, "ziwei", "hospital")

