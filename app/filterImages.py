import csv
import os
from random import random
import pandas as pd


def split_data():
    df = pd.read_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-v2.csv")
    df_test = pd.read_csv("D:/Licenta/Proiect/data/BCS-DBT file-paths-v2.csv")
    df_label_train = pd.read_csv("D:/Licenta/Proiect/data/labels/BCS-DBT labels-v2.csv")
    df_label_test = pd.read_csv("D:/Licenta/Proiect/data/labels/BCS-DBT labels-v2.csv")
    threshold = df.shape[0]
    train_test_proportion = 0.8
    for i in range(0, threshold):
        accepted = random()
        if accepted > train_test_proportion:
            df = df.drop(i)
            df_label_train = df_label_train.drop(i)
        else:
            df_test = df_test.drop(i)
            df_label_test = df_label_test.drop(i)
    df.to_csv("BCS-DBT file-paths-train-v2.csv")
    df_test.to_csv("BCS-DBT file-paths-test-v2.csv")
    df_label_train.to_csv("labels/BCS-DBT labels-train-v2.csv")
    df_label_test.to_csv("labels/BCS-DBT labels-test-v2.csv")


def clean_files(split_name):
    new_rows = ""
    with open(f"D:/Licenta/Proiect/data/BCS-DBT file-paths-{split_name}-v2.csv", 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            new_rows = new_rows + row[1]
            for column in row[2:]:
                new_rows = new_rows + ',' + column
            new_rows = new_rows + '\n'
    with open(f"D:/Licenta/Proiect/data/BCS-DBT file-paths-{split_name}-v2.csv", 'w+') as writeTo:
        writeTo.write(new_rows[:-1])


def keep_rmlo_views():
    df = pd.read_csv("data/BCS-DBT file-paths-v2.csv")
    for study in df.iloc:
        view_series = study
        view = view_series["View"]
        if view != "rmlo":
            image_path = os.path.join("data/images/manifest-1617905855234/", view_series["descriptive_path"])
            if os.path.exists(image_path):
                os.remove(image_path)
                os.rmdir(
                    image_path.split("/")[0] + "/" + image_path.split("/")[1] + "/" + image_path.split("/")[2] + "/" +
                    image_path.split("/")[3] + "/" + image_path.split("/")[4] + "/" + image_path.split("/")[5])
