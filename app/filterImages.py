import csv
import os
from random import random
import pandas as pd

def splitData():
    df = pd.read_csv("data/BCS-DBT file-paths-v2.csv")
    dfTest = pd.read_csv("data/BCS-DBT file-paths-v2.csv")
    dfLabelTrain = pd.read_csv("data/labels/BCS-DBT labels-v2.csv")
    dfLabelTest = pd.read_csv("data/labels/BCS-DBT labels-v2.csv")
    threshold = df.shape[0]
    train_test_proportion = 0.8
    for i in range(0, threshold):
        accepted = random()
        if accepted > train_test_proportion:
            df = df.drop(i)
            dfLabelTrain = dfLabelTrain.drop(i)
        else:
            dfTest = dfTest.drop(i)
            dfLabelTest = dfLabelTest.drop(i)
        print(df.iloc[i]["PatientID"])
    df.to_csv("BCS-DBT file-paths-train-v2.csv")
    dfTest.to_csv("BCS-DBT file-paths-test-v2.csv")
    dfLabelTrain.to_csv("labels/BCS-DBT labels-train-v2.csv")
    dfLabelTest.to_csv("labels/BCS-DBT labels-test-v2.csv")

def cleanFiles():
    new_Rows = ""
    with open("data/BCS-DBT file-paths-train-v2.csv", 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            new_Rows = new_Rows + row[1]
            for column in row[2:]:
                new_Rows = new_Rows + ',' + column
            new_Rows = new_Rows + '\n'
    with open("data/BCS-DBT file-paths-train-v2.csv", 'w+') as writeTo:
        writeTo.write(new_Rows[:-1])

def keepRMLOViews():
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
