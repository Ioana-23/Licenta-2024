import csv
import os
from random import random

import cv2
import pandas as pd
from duke_dbt_data import dcmread_image
import shutil
import numpy as np
from skimage.io import imsave


# Splits all the data into train and test data with a 80% and 20% spread
def split_data():
    df = pd.read_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-all.csv")
    df_test = pd.read_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-all.csv")
    df_label_train = pd.read_csv("D:/Licenta/Proiect/data/labels/BCS-DBT labels-all.csv")
    df_label_test = pd.read_csv("D:/Licenta/Proiect/data/labels/BCS-DBT labels-all.csv")
    df_boxes_train = pd.read_csv("D:/Licenta/Proiect/data/bounding_boxes/BCS-DBT boxes-all.csv")
    df_boxes_test = pd.read_csv("D:/Licenta/Proiect/data/bounding_boxes/BCS-DBT boxes-all.csv")
    threshold = df.shape[0]
    train_test_proportion = 0.8
    index_aux_train = 0
    index_aux_test = 0
    for i in range(0, threshold):
        accepted = random()
        if accepted > train_test_proportion:
            index = df_boxes_train.index[
                df_boxes_train["PatientID"] == df_label_train.iloc[index_aux_train]["PatientID"]].tolist()
            if len(index) != 0:
                df_boxes_train.drop(index, inplace=True)
            df.drop(i, inplace=True)
            df_label_train.drop(i, inplace=True)
            index_aux_train = index_aux_train - 1
        else:
            index = df_boxes_test.index[
                df_boxes_test["PatientID"] == df_label_test.iloc[index_aux_test]["PatientID"]].tolist()
            if len(index) != 0:
                df_boxes_test.drop(index, inplace=True)
            df_test.drop(i, inplace=True)
            df_label_test.drop(i, inplace=True)
            index_aux_test = index_aux_test - 1
        index_aux_test = index_aux_test + 1
        index_aux_train = index_aux_train + 1
    df.to_csv("BCS-DBT file-paths-train.csv", index=False)
    df_test.to_csv("BCS-DBT file-paths-test.csv", index=False)
    df_label_train.to_csv("labels/BCS-DBT labels-train.csv", index=False)
    df_label_test.to_csv("labels/BCS-DBT labels-test.csv", index=False)
    df_boxes_train.to_csv("bounding_boxes/BCS-DBT boxes-train.csv", index=False)
    df_boxes_test.to_csv("bounding_boxes/BCS-DBT boxes-test.csv", index=False)


# Iterates through the image directory and deletes all the scans not made from the 'rmlo' position
# Deletes only the images, not the rows in which the images were found
def keep_rmlo_views():
    df = pd.read_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-all.csv")
    for study in df.iloc:
        view_series = study
        view = view_series["View"]
        if view != "rmlo":
            image_path = find_image_path(view_series)
            dir_path = "D:/Licenta/Proiect/data/images"
            for i in range(5, 9):
                dir_path = dir_path + "/" + image_path.split("/")[i]
            if os.path.exists(image_path):
                os.remove(image_path)
            while os.path.exists(dir_path) and len(os.listdir(dir_path)) == 0:
                os.rmdir(dir_path)
                index = 0
                for i in range(0, len(dir_path)):
                    if dir_path[i] == "/":
                        index = i
                dir_path = dir_path[:index]


# Iterates through all the images and finds the minimum number of slices across all scans
def find_minimum_slices():
    breast_cancer_data = pd.read_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-all.csv")
    threshold = breast_cancer_data.shape[0]
    max_number_of_slices = 22
    for i in range(722, threshold):
        view_series = breast_cancer_data.iloc[i]
        view = view_series["View"]
        image_path = find_image_path(view_series)

        image = dcmread_image(fp=image_path, view=view)
        max_number_of_slices = min(max_number_of_slices, image.shape[0])
        print(str(i) + "::" + view_series["PatientID"] + '/' + view_series["StudyUID"])
        if (i + 1) % 100 == 0:
            print('Went over 100 more labels!')
            print(max_number_of_slices)
    print('Went over all the labels!')
    print(max_number_of_slices)


def find_image_path(base_folder, view_series):
    image_path = os.path.join(base_folder, view_series["descriptive_path"])
    final_image = ""
    for i in range(0, len(image_path.split("-")) - 2):
        final_image = final_image + image_path.split("-")[i] + '-'
    final_image = final_image + 'NA'
    for i in range(len(image_path.split("-")) - 2, len(image_path.split("-"))):
        final_image = final_image + "-" + image_path.split("-")[i]
    image_path = final_image
    return image_path[:image_path.rindex("/")+1]


# Sorts the .csv files after PatientID and StudyUID
def sort_files(filename):
    df = pd.read_csv(f"D:/Licenta/Proiect/data/bounding_boxes/{filename}.csv")
    df.sort_values(["PatientID", "StudyUID"], axis=0, ascending=[True, True], inplace=True)
    df.to_csv(f"{filename}.csv", index=False)


# Deletes the images that do not correspond between what is in the subdirectories and
# what is in the true label file
def remove_surplus_images():
    dir_path = "D:/Licenta/Proiect/data/images/Breast-Cancer-Screening-DBT"
    dir_list = os.listdir(dir_path)
    file = pd.read_csv("../Project/app/data/labels/BCS-DBT labels-all.csv")
    for directory in dir_list:
        if directory in file["PatientID"].tolist():
            directory_list = os.listdir(os.path.join(dir_path, directory))
            for sub_directory in directory_list:
                study_name = "DBT" + sub_directory.split("DBT")[1][:7]
                if study_name not in file["StudyUID"].tolist():
                    shutil.rmtree(os.path.join(os.path.join(dir_path, directory), sub_directory))
        else:
            shutil.rmtree(os.path.join(dir_path, directory))


# Deletes the rows that do not correspond between the file_path and the true label files
def get_appropriate_studies():
    df_label, df_file_path = pd.read_csv("D:/Licenta/Proiect/data/labels/BCS-DBT labels-all.csv"), \
        pd.read_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-all.csv")
    index = -1
    for i in range(0, df_file_path.shape[0]):
        index = index + 1
        study = df_file_path.iloc[index]
        if study["View"] != "rmlo":
            df_file_path.drop(i, axis=0, inplace=True)
            index = index - 1
        else:
            if study["PatientID"] in df_label["PatientID"].tolist():
                if study["StudyUID"] not in df_label["StudyUID"].tolist():
                    df_file_path.drop(i, axis=0, inplace=True)
                    index = index - 1
            else:
                df_file_path.drop(i, axis=0, inplace=True)
                index = index - 1
    df_file_path.to_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-all.csv", index=False)


# Verifies that the contents of the file_path and true label files are the same,
# printing "Problem" if the other fits
def verify_files():
    df_label = pd.read_csv("../Project/app/data/labels/BCS-DBT labels-all.csv")
    dir_path = "D:/Licenta/Proiect/data/images/Breast-Cancer-Screening-DBT"
    dir_list = os.listdir(dir_path)
    i, j = 0, 0
    while i < df_label.shape[0] or j < len(dir_list) - 1:
        patient_id = df_label.iloc[i]["PatientID"]
        if patient_id == dir_list[j]:
            directories = os.listdir(os.path.join(dir_path, dir_list[j]))
            i_copy = i
            for directory in directories:
                study_id = df_label.iloc[i_copy]["StudyUID"]
                if study_id[3:] != directory.split("DBT")[1][:7]:
                    print("Problem")
                i_copy = i_copy + 1
            i = i + len(directories) - 1
        else:
            print("Here")
        i = i + 1
        j = j + 1


def save_slices_to_png(f_path, dcm_path, index, number_of_slices):
    image = np.array(dcmread_image(fp=os.path.join(dcm_path, "1-1.dcm"), view="rmlo"))
    image = image.astype(np.float32) * 255. / image.max()
    image = image.astype(np.uint8)
    if index == -1:
        index = image.shape[0] // 2
    min_lim = max(0, index - number_of_slices // 2)
    max_lim = min(index + number_of_slices // 2, image.shape[0])
    if min_lim == 0:
        print()
        to_add = (index - number_of_slices // 2) * -1
        max_lim = max_lim + to_add
    if max_lim == image.shape[0]:
        print()
        to_add = image.shape[0] - (index + number_of_slices // 2)
        min_lim = min_lim - to_add
    slices = image[min_lim:max_lim, :, :]
    for i in range(number_of_slices):
        new_image = cv2.resize(slices[i], (512, 512))
        imsave(os.path.join(f_path, f"{min_lim + i}.png"), new_image)
    open(os.path.join(f_path, f"{image.shape}.txt"), 'a')
    #imsave(os.path.join(f_path, "0.png"), image[0, :, :1890])


def save_all_slices(split_name):
    base_folder = "D:/Licenta/Project/app/data/"
    breast_cancer_data = pd.read_csv(os.path.join(base_folder, f"BCS-DBT file-paths-{split_name}.csv"))
    breast_cancer_boxes = pd.read_csv(os.path.join(base_folder,
                                                   f"bounding_boxes/BCS-DBT boxes-{split_name}.csv"))
    for idx in range(breast_cancer_data.shape[0]):
        view_series = breast_cancer_data.iloc[idx]
        image_path = find_image_path("D:/Licenta/Proiect/data/images/", view_series)
        dcm_path = find_image_path(os.path.join(base_folder, "images"), view_series)
        patient_id = breast_cancer_data.iloc[idx]["PatientID"]
        index = breast_cancer_boxes.index[breast_cancer_boxes["PatientID"] == patient_id].tolist()
        slice_index = -1
        if len(index) != 0:
            slice_index = int(breast_cancer_boxes.iloc[index[0]]["Slice"])
        save_slices_to_png(image_path, dcm_path, slice_index, 22)


save_all_slices("all")

