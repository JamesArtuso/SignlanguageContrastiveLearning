import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class ContrastiveLearningDataset(Dataset):
    def __init__(self, data_dir, negative_size = 10, transform=None):
        """
        Custom Dataset for contrastive learning.
        
        Args:
            data_dir (str): Path to the directory containing numpy files for each class.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load all the class data (each file corresponds to one class)
        self.class_data = {}
        self.class_images = {}
        for file in os.listdir(data_dir):
            if file.endswith('.npy'):
                class_id = int(file.split('.')[0].split('_')[1])  # Assuming file is like class_n.npy
                self.class_data[class_id] = np.load(os.path.join(data_dir, file), allow_pickle = True)[0]
                self.class_images[class_id] = np.load(os.path.join(data_dir, file), allow_pickle= True)[1]
        
        # Store class ids in a list
        self.classes = list(self.class_data.keys())
        self.negative_size = negative_size

    def __len__(self):
        # Total number of samples is the sum of all samples in each class
        return sum(len(data) for data in self.class_data.values())

    def __getitem__(self, idx):
        # Randomly pick a class
        class_id = random.choice(self.classes)
        class_samples = self.class_data[class_id]
        image_samples = self.class_images[class_id]
        
        # Randomly pick a sample from the selected class
        anchor_idx = random.randint(0, len(class_samples) - 1)
        anchor = class_samples[anchor_idx,:,:]
        anchor_image = image_samples[anchor_idx,:,:,:]
        
        # Positive sample (same class)
        pos_idx = random.randint(0, len(class_samples) - 1)
        positive_sample = class_samples[pos_idx,:,:]
        #positive_image = image_samples[pos_idx,:,:,:]
        positive_image = np.zeros((10,10,10,10))

        # Negative sample (different class)
        negative_class_id = np.random.choice([c for c in self.classes if c != class_id],size=(self.negative_size), replace=True)
        neg_idx = [np.random.randint(0, len(self.class_data[i]) - 1) for i in negative_class_id]
        negative_samples = np.array([self.class_data[negative_class_id[i]][neg_idx[i],:,:] for i in range(self.negative_size)])
        #negative_images = np.array([self.class_images[negative_class_id[i]][neg_idx[i],:,:,:] for i in range(self.negative_size)])
        negative_images = np.zeros((10,10,10,10))
        # Apply any transformations (optional)
        if self.transform:
            anchor = self.transform(anchor)
            positive_sample = self.transform(positive_sample)
            negative_sample = self.transform(negative_sample)
        
        # Return the anchor, positive sample, and negative sample
        return anchor, anchor_image, positive_sample, positive_image, negative_samples, negative_images, class_id


class ClassificationLearningDataset(Dataset):
    def __init__(self, data_dir, model, device, transform=None):
        """
        Custom Dataset for contrastive learning.
        
        Args:
            data_dir (str): Path to the directory containing numpy files for each class.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data_dir = data_dir
        self.model = model.eval()
        
        # Load all the class data (each file corresponds to one class)
        self.class_data = []
        self.class_ids = []
        for file in os.listdir(data_dir):
            if file.endswith('.npy'):
                class_id = int(file.split('.')[0].split('_')[1])  # Assuming file is like class_n.npy
                with(torch.no_grad()):
                    data = self.model(torch.tensor(np.load(os.path.join(data_dir, file), allow_pickle = True)[0]).float().to(device)).cpu()
                for j in range(data.shape[0]):
                    self.class_data.append(data[j,:])
                    self.class_ids.append(class_id)

    def __len__(self):
        # Total number of samples is the sum of all samples in each class
        return len(self.class_data)

    def __getitem__(self, idx):
        return self.class_data[idx], self.class_ids[idx]


class EvaluationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Custom Dataset for contrastive learning.
        
        Args:
            data_dir (str): Path to the directory containing numpy files for each class.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load all the class data (each file corresponds to one class)
        self.class_data = []
        self.class_ids = []
        for file in os.listdir(data_dir):
            if file.endswith('.npy'):
                class_id = int(file.split('.')[0].split('_')[1])  # Assuming file is like class_n.npy
                data = np.load(os.path.join(data_dir, file), allow_pickle = True)[0]
                for j in range(data.shape[0]):
                    self.class_data.append(data[j,:,:])
                    self.class_ids.append(class_id)

    def __len__(self):
        # Total number of samples is the sum of all samples in each class
        return len(self.class_data)

    def __getitem__(self, idx):
        return self.class_data[idx], self.class_ids[idx]
'''
# Usage example:
data_dir = "/path/to/numpy_files"  # Change this to your numpy files directory
dataset = ContrastiveLearningDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for anchor, positive, negative in dataloader:
    # Example: Perform contrastive loss calculation
    pass
'''
