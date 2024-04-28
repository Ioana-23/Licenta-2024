import os
import numpy as np
import pandas as pd
import torch.utils.data
from skimage.io import imread
import torchvision.transforms.functional as fn
import albumentations as A
from app.utils.sort_split_data import find_image_path


def get_train_transforms():
    return A.Compose([
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.9),
        ], p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels'],
        )
    )


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


class DetectionDataset(torch.utils.data.Dataset):

    def __init__(self, base_folder, split_name, number_of_slices, batch_size,
                 transfm=get_train_transforms(), threshold=None):
        super().__init__()
        self.transfm = transfm
        self.base_folder = base_folder
        self.batch_size = batch_size
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
            self.threshold = self.breastCancerBoxes.shape[0]
        self.targets = []
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.load_data()

    def calculate_class_indices(self):
        class_indices = {}
        for class_label in range(2):
            class_indices[class_label] = np.where(np.array(self.targets) == class_label)[0]
        return class_indices

    def load_image(self, path, file_name):
        slices = np.zeros(shape=[3, 512, 512], dtype=np.float32)
        slices[0] = imread(fname=os.path.join(path, file_name), as_gray=True)
        slices[0] = slices[0] / 255.0
        slices = torch.tensor(slices)
        slices[1] = slices[0]
        slices[2] = slices[0]
        slices = fn.normalize(slices, mean=self.mean, std=self.std)
        return slices

    def load_data(self):
        minimum = 104
        for idx in range(0, self.threshold):
            view_series = self.breastCancerBoxes.iloc[idx]
            label = view_series["Class"]
            if label == 'benign':
                label = 0
            else:
                label = 1
            index_fisier = self.breastCancerData.index[
                self.breastCancerData["PatientID"] == view_series["PatientID"]].tolist()[0]
            image_path = find_image_path(self.breastCancerData.iloc[index_fisier], os.path.join(self.base_folder, "images"))
            shape = get_nth_file_name(image_path, 0)
            slices = int(shape[1:shape.index(",")])
            if slices < minimum:
                minimum = slices
            for i in range(self.number_of_slices):
                self.targets.append(label)
                if image_path[-1] == "/":
                    self.data_paths.append(image_path + f"{i + (10 - self.number_of_slices // 2)}")
                else:
                    self.data_paths.append(image_path + f"/{i + (10 - self.number_of_slices // 2)}")

        if self.batch_size == 1:
            return

        class_indices = self.calculate_class_indices()
        batches = []
        malign = len(class_indices[1]) - len(class_indices[0])
        if malign > 0:
            sampled_indices = np.random.choice(class_indices[1], size=malign, replace=False)
            indices = []
            for sample in sampled_indices:
                index = np.where(class_indices[1] == sample)
                indices.append(int(index[0][0]))
            class_indices[1] = np.delete(class_indices[1], indices)
            self.data_paths = np.delete(self.data_paths, sampled_indices)
            self.targets = np.delete(self.targets, sampled_indices)
            class_indices = self.calculate_class_indices()
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

    def __len__(self):
        return len(self.data_paths)

    def scale_coordinates(self, index):
        image_path = find_image_path(self.breastCancerData.iloc[index], os.path.join(self.base_folder, "images"))
        shape = get_nth_file_name(image_path, 0)
        original_width = int(shape[shape.index(",") + 1:shape.rindex(",")])
        original_height = int(shape[shape.rindex(",") + 1:shape.rindex(")")])

        new_width = 512
        new_height = 512

        scale_width = new_width / original_width
        scale_height = new_height / original_height

        return scale_width, scale_height

    def load_bounding_box(self, idx, index_fisier):
        scaled_width, scaled_height = self.scale_coordinates(index_fisier)
        x, y, width, height = self.breastCancerBoxes.iloc[idx]["X"], \
            self.breastCancerBoxes.iloc[idx]["Y"], \
            self.breastCancerBoxes.iloc[idx]["Width"], \
            self.breastCancerBoxes.iloc[idx]["Height"]
        boxes = [x * scaled_height, y * scaled_width,
                 (x + width) * scaled_height, (y + height) * scaled_width]
        label = self.breastCancerBoxes.iloc[idx]["Class"]
        if label == 'benign':
            label = 1.
        else:
            label = 2.
        boxes = torch.tensor(boxes, dtype=torch.float32).unsqueeze(dim=0)
        label = torch.tensor(label).unsqueeze(dim=0)

        target = {'bbox': boxes, 'cls': label}

        return target

    def __getitem__(self, index):
        slice_path = self.data_paths[index]
        directory_path = slice_path[:slice_path.rindex("/") + 1]
        path_in_file = slice_path[len(self.base_folder) + 7:]
        path_in_file = path_in_file[:path_in_file.rindex("/") + 1] + "1-1.dcm"
        file_name_1 = path_in_file[:path_in_file.rindex("NA") - 1]
        path_in_file = file_name_1 + path_in_file[path_in_file.rindex("NA") + 2:]
        index_fisier = self.breastCancerData.index[
            self.breastCancerData["descriptive_path"] == path_in_file].tolist()[0]
        index_fisier_boxes = self.breastCancerBoxes.index[
            self.breastCancerBoxes["PatientID"] == self.breastCancerData.iloc[index_fisier]["PatientID"]].tolist()[0]
        index_slice = slice_path[self.data_paths[index].rindex("/") + 1]
        file_name = get_nth_file_name(directory_path, int(index_slice) + 1)
        image = self.load_image(directory_path, file_name)
        bounding_box = self.load_bounding_box(index_fisier_boxes, index_fisier)

        sample = self.transfm(**{
            'image': np.array(image.permute(1, 2, 0)),
            'bboxes': bounding_box['bbox'],
            'labels': bounding_box['cls']
        })
        target = {}
        image = torch.tensor(sample['image']).permute(2, 0, 1)
        target['bbox'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        target['bbox'][:, [0, 1, 2, 3]] = target['bbox'][:, [1, 0, 3, 2]]
        target['bbox'] = target['bbox'].clone().detach()
        target['bbox'] = target['bbox'].to(torch.float32).cpu()
        target['cls'] = torch.stack(sample['labels']).cpu()
        return image.cpu(), target