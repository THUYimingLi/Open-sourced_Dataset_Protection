import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None, y_target=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)
        self.transform = transform
        if y_target is not None:
            self.csv_data.iloc[:, 1] = y_target

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId



class GTSRB_subset(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None, List=[], y_target=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            List: the index of selected sample idxs
        """
        assert len(List) > 0, "Dataset should contain at least one sample"

        self.root_dir = root_dir
        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path).iloc[List]
        self.transform = transform
        if y_target is not None:
            self.csv_data.iloc[:, 1] = y_target


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId



class GTSRB_subclass(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None, Class=2, y_target=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            Class: the selected class
        """
        assert len(List) > 0, "Dataset should contain at least one sample"

        self.root_dir = root_dir
        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'
        self.Class = Class
        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        All_data = pd.read_csv(csv_file_path)
        List = [i for i in range(len(All_data)) if All_data.iloc[i, 1] == self.Class]
        self.csv_data = pd.read_csv(csv_file_path).iloc[List]
        self.transform = transform
        if y_target is not None:
            self.csv_data.iloc[:, 1] = y_target


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


class GTSRB_Testset(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None, select_class=2, num_img=100):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            selected class: the class of selected samples
            num_img: number of selected images
        """

        self.root_dir = root_dir
        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)
        self.csv_data_new = pd.DataFrame(columns=['Filename', 'ClassId'])

        for i in range(len(self.csv_data)):
            if self.csv_data.iloc[i, 1] == select_class:
                self.csv_data_new = self.csv_data_new.append(self.csv_data.iloc[i])
        # randomly idx
        random.seed(random.randint(1, 10000))
        idx = list(np.arange(len(self.csv_data_new)))
        random.shuffle(idx)
        image_idx = idx[:num_img]

        self.csv_data_final = self.csv_data_new.iloc[image_idx]  # final data
        self.transform = transform

    def __len__(self):
        return len(self.csv_data_final)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data_final.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data_final.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId