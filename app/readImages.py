import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data
from torchvision.transforms import v2
from duke_dbt_data import dcmread_image


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, base_folder, transforms, threshold, split_name):
        super().__init__()
        self.number_of_slices = 22
        self.transforms = transforms
        self.base_folder = base_folder
        self.threshold = threshold
        self.images_file = os.path.join(self.base_folder, "images/manifest-1617905855234/")
        self.label_file = os.path.join(self.base_folder, f"labels/BCS-DBT labels-{split_name}-v2.csv")
        self.data_paths = []
        self.breastCancerData = pd.read_csv(os.path.join(self.base_folder, f"BCS-DBT file-paths-{split_name}-v2.csv"))
        self.breastCancerLabel = pd.read_csv(self.label_file)
        self.load_data()
        self.biggest_value_for_color = 0

    def load_data(self):
        if self.threshold is not None:
            for idx in range(0, self.threshold):
                view_series = self.breastCancerData.iloc[idx]
                self.data_paths.append(os.path.join(self.images_file, view_series["descriptive_path"]))
        else:
            for series in self.breastCancerData.iloc:
                self.data_paths.append(os.path.join(self.images_file, series["descriptive_path"]))

    def __len__(self):
        return len(self.data_paths)

    def load_label(self, idx):
        return np.array(self.breastCancerLabel.iloc[idx][3:].values, dtype=np.float32)

    def load_image(self, path):
        slices = dcmread_image(fp=path, view="rmlo")
        middle_of_slices = slices.shape[0] // 2
        middle_of_image = self.number_of_slices // 2
        slices = np.array(slices[middle_of_slices - middle_of_image: middle_of_slices + middle_of_image],
                          dtype=np.float32).transpose((1, 2, 0))
        return np.divide(slices, 65535)

    def __getitem__(self, index):
        image = self.load_image(self.data_paths[index])
        label = self.load_label(index)
        if self.transforms is not None:
            image = self.transforms(image)
        self.biggest_value_for_color = image.max()
        return image, label


# trainDataset = MyDataset(base_folder="D:/Licenta/Proiect/data/", transforms=v2.Compose([v2.Resize(1996),
#                                                                                         v2.CenterCrop(1890),
#                                                                                         v2.ToTensor()]
#                                                                                        ),
#                          threshold=2, split_name="train")
# testDataset = MyDataset(base_folder="D:/Licenta/Proiect/data/", transforms=v2.Compose([v2.Resize(1996),
#                                                                                        v2.CenterCrop(1890),
#                                                                                        v2.ToTensor()]
#                                                                                       ),
#                         threshold=2, split_name="test")
# print("Number of samples in dataset: ", len(trainDataset))
# print("Number of samples in dataset: ", len(testDataset))
#
# bs = 2
# trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=bs, shuffle=True, num_workers=0)
# testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=bs, shuffle=True, num_workers=0)
#
# for i_batch, sample_batched in enumerate(trainDataloader):
#     imgs = sample_batched[0]
#     segs = sample_batched[1]
#
#     rows, cols = bs, 2
#     figure = plt.figure(figsize=(10, 10))
#
#     for i in range(0, bs):
#         figure.add_subplot(rows, cols, 2 * i + 1)
#         plt.title('image')
#         plt.axis("off")
#         plt.imshow(imgs[i].cpu().numpy()[0])
#
#         print(segs[i])
#     plt.show()
#     if i_batch == 2:
#         break

# allImages = []
#
# threshold = breastCancerData.shape[0]
# firstImagesShape = []
# maxNumberOfSlices = 22
# for i in range(1819, threshold):
#     view_series = breastCancerData.iloc[i]
#     view = view_series["View"]
#     image_path = os.path.join("images/manifest-1617905855234/", view_series["descriptive_path"])
#
#     image = dcmread_image(fp=image_path, view=view)
#     maxNumberOfSlices = min(maxNumberOfSlices, image.shape[0])
#     # path = 'D:/Licenta/Proiect/data/trainImages'
#     # try:
#     #     os.mkdir(path)
#     # except OSError as error:
#     #     pass
#     # path = path + '/' + view_series["PatientID"]
#     # try:
#     #     os.mkdir(path)
#     # except OSError as error:
#     #     pass
#     # path = path + '/' + view_series["StudyUID"]
#     # try:
#     #     os.mkdir(path)
#     # except OSError as error:
#     #     pass
#     # for j in range(0, image.shape[0]):
#     #     np.save(path + '/file' + str(j) + '.npy', image[j])
#     print(view_series["PatientID"] + '/' + view_series["StudyUID"])
#     if i % 100 == 0:
#         print('Went over 50 more labels!')
#         print(maxNumberOfSlices)
#     # firstImagesShape.append(image.shape)
#     # blankImage = np.zeros((1, 2457, 1890), dtype=np.uint8)
#     # firstImagesMesh = np.array([slice for slice in image])
#     # firstImagesMesh = np.append(blankImage, firstImagesMesh, axis=0)
#     # firstImagesMesh = np.append(firstImagesMesh, blankImage, axis=0)
#     # print(firstImagesMesh.shape)
#     # verts, faces, _, _ = measure.marching_cubes(firstImagesMesh, 0.0)
#     # obj_3D = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
#     # for i, f in enumerate(faces):
#     #     obj_3D.vectors[i] = verts[f]
#     # obj_3D.save('mesh.stl')
#     # plt.imshow(image[0], cmap=plt.cm.gray)
#     # plt.show()


# def viewSTL():
#     reader = vtk.vtkSTLReader()
#     reader.SetFileName('mesh.stl')
#
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputConnection(reader.GetOutputPort())
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#
#     renderer = vtk.vtkRenderer()
#     rendererWindow = vtk.vtkRenderWindow()
#     rendererWindow.AddRenderer(renderer)
#     rendererWindowInteractor = vtk.vtkRenderWindowInteractor()
#     rendererWindowInteractor.SetRenderWindow(rendererWindow)
#
#     renderer.AddActor(actor)
#     renderer.SetBackground(0, 0, 0)
#
#     rendererWindow.Render()
#     rendererWindowInteractor.Start()
# viewSTL()
