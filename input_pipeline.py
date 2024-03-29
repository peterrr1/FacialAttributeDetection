import glob
from PIL import Image
import sys
import pandas as pd
import torch
import torchvision
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
from model import MobileNet

PATH_TO_IMAGES = 'data/img_align_celeba'
PATH_TO_LABELS = 'data/list_attr_celeba.csv'


class ImageLoader(Dataset):
    def __init__(self, data_path, label_path, img_size=(234, 234)):

        self.data_path = data_path
        self.label_path = label_path
        self.images = self.get_images_from_directory()
        self.labels = self.get_labels_from_csv()
        self.attr_names = self.get_attribute_names_from_csv()

        self.transform = v2.Compose([
            v2.Resize(size=img_size),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # Normalization for pretrained mobilenet: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image_tensor = self.transform(image)
        label_tensor = torch.Tensor(self.labels[idx])
        return image_tensor, label_tensor

    def __len__(self):
        return len(self.images)

    def get_images_from_directory(self):
        return sorted(glob.glob(f'{self.data_path}/*.jpg'))

    def get_labels_from_csv(self):
        label_list = open(self.label_path).readlines()[1:]
        data_label = []
        for i in range(len(label_list)):
            data_label.append(label_list[i].strip().split(',')[1:])
        for i in range(len(data_label)):
            data_label[i] = [j.replace('-1', '0') for j in data_label[i]]
            data_label[i] = [int(j) for j in data_label[i]]
        return data_label

    def get_attribute_names_from_csv(self):
        return open(self.label_path).readlines()[0].split(',')[1:]



def create_dataset_split(dataset, batch_size=64):

    indices = list(range(len(dataset)))
    # train split is 70%
    train_split = int(len(indices) * 0.7)

    # validation split is 20%
    valid_split = int(len(indices) * 0.9)

    # test split is 10%
    train_idx, valid_idx, test_idx = indices[:train_split], indices[train_split:valid_split], indices[valid_split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_data = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_data = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_data = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_data, valid_data, test_data




if __name__ == '__main__':
    dataset = ImageLoader(PATH_TO_IMAGES, PATH_TO_LABELS)
    train_data, valid_data, test_data = create_dataset_split(dataset)

    model = MobileNet()
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    #model.fit(train_data, optimizer, loss_fn, 5)

    