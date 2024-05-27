import os
import numpy as np
import pandas as pd
import torch.utils.data
from skimage.io import imread
import torchvision.transforms.functional as fn
from app.system_utils.sort_split_data import find_image_path


def get_nth_file_name(directory, n):
    return os.listdir(directory)[n]


def extract_indices(batch, index, size, class_indices):
    sampled_indices = np.random.choice(class_indices[index], size=size, replace=False)
    batch.extend(sampled_indices.tolist())
    indices = []
    for sample in sampled_indices:
        new_index = np.where(class_indices[index] == sample)
        indices.append(int(new_index[0][0]))
    indices = np.array(indices)
    class_indices[index] = np.delete(class_indices[index], indices)


def add_class(batch, index, amount, class_indices):
    extract_indices(batch, index, amount, class_indices)


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, base_folder, split_name, number_of_slices, batch_size, label_min=0,
                 threshold=None):
        super().__init__()
        self.label_min = label_min
        self.batch_size = batch_size
        self.base_folder = base_folder
        self.number_of_slices = number_of_slices
        self.threshold = threshold
        self.images_file = os.path.join(self.base_folder, "images/")
        self.label_file = os.path.join(self.base_folder, f"labels/BCS-DBT labels-{split_name}.csv")
        self.data_paths = []
        self.breastCancerData = pd.read_csv(os.path.join(self.base_folder, f"BCS-DBT file-paths-{split_name}.csv"))
        self.breastCancerBoxes = pd.read_csv(os.path.join(self.base_folder,
                                                          f"bounding_boxes/BCS-DBT boxes-{split_name}.csv"))
        self.breastCancerLabel = pd.read_csv(self.label_file)
        if threshold is None:
            self.threshold = self.breastCancerData.shape[0]
        self.targets = []
        self.load_data()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_image(self, path, file_name):
        slices = np.empty(shape=[3, 512, 512], dtype=np.float32)
        slices[0] = imread(fname=os.path.join(path, file_name), as_gray=True)
        slices[0] = slices[0] / 255.0
        slices = torch.tensor(slices)
        slices[1] = slices[0]
        slices[2] = slices[0]
        slices = fn.normalize(slices, mean=self.mean, std=self.std)
        return np.array(slices)

    def calculate_class_indices(self):
        class_indices = {}
        for class_label in range(2):
            class_indices[class_label] = np.where(np.array(self.targets) == class_label)[0]
        return class_indices

    def load_data(self):
        for idx in range(0, self.threshold):
            view_series = self.breastCancerData.iloc[idx]
            label = np.argmax(self.breastCancerLabel.iloc[idx][3:].values)
            if label < self.label_min:
                continue
            if label != self.label_min:
                label = 1
            else:
                label = 0
            image_path = find_image_path(view_series=view_series, base_folder=os.path.join(self.base_folder, "images"))
            self.add_corresponding_number_of_slices(image_path, label)
        if self.batch_size == 1:
            return

        self.get_even_batches()

    def add_corresponding_number_of_slices(self, image_path, label):
        for i in range(self.number_of_slices):
            self.targets.append(label)
            if image_path[-1] == "/":
                self.data_paths.append(image_path + f"{i}")
            else:
                self.data_paths.append(image_path + f"/{i}")

    def get_even_batches(self):
        class_indices = self.calculate_class_indices()
        batches = []
        class_indices = self.level_inputs(class_indices)
        batches_number = len(self) // self.batch_size
        if batches_number * self.batch_size != len(self):
            batches_number = batches_number + 1
        batch_number = 0
        while batch_number < batches_number:
            batch = []
            add_class(batch, 1, min(self.batch_size // 2, len(class_indices[1])), class_indices)
            add_class(batch, 0, min(self.batch_size // 2, len(class_indices[0])), class_indices)
            np.random.shuffle(batch)
            batches.extend(batch)
            batch_number = batch_number + 1
        self.data_paths = np.array(self.data_paths)
        batches = np.array(batches)
        self.data_paths = self.data_paths[batches]

    def level_inputs(self, class_indices):
        index = 0
        class_indices = self.level_one_input(class_indices, index)
        class_indices = self.level_one_input(class_indices, index+1)
        return class_indices

    def level_one_input(self, class_indices, index):
        type_allowed = len(class_indices[index]) - len(class_indices[(index+1) % 2])
        if type_allowed > 0:
            sampled_indices = np.random.choice(class_indices[index], size=type_allowed, replace=False)
            indices = []
            for sample in sampled_indices:
                indexes = np.where(class_indices[index] == sample)
                indices.append(int(indexes[0][0]))
            class_indices[index] = np.delete(class_indices[index], indices)
            self.data_paths = np.delete(self.data_paths, sampled_indices)
            self.targets = np.delete(self.targets, sampled_indices)
            class_indices = self.calculate_class_indices()
        return class_indices

    def __len__(self):
        return len(self.data_paths)

    def load_label(self, idx):
        array = np.array(self.breastCancerLabel.iloc[idx][3:].values, dtype=np.float32)
        if array[self.label_min] != 1:
            return np.array((0, 1))
        return np.array((1, 0))

    def __getitem__(self, index):
        slice_path = self.data_paths[index]
        directory_path = slice_path[:slice_path.rindex("/") + 1]
        path_in_file = slice_path[len(self.base_folder) + 7:]
        path_in_file = str(path_in_file[:path_in_file.rindex("/") + 1]) + "1-1.dcm"
        file_name_1 = path_in_file[:path_in_file.rindex("NA") - 1]
        path_in_file = file_name_1 + path_in_file[path_in_file.rindex("NA") + 2:]
        index_fisier = self.breastCancerData.index[
            self.breastCancerData["descriptive_path"] == path_in_file].tolist()[0]
        index_slice = slice_path[self.data_paths[index].rindex("/") + 1]
        file_name = get_nth_file_name(directory_path, int(index_slice) + 1)
        image = self.load_image(directory_path, file_name)
        label = self.load_label(index_fisier)
        return image, label
