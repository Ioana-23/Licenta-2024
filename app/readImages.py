import os
import matplotlib.pyplot as plt
import pandas as pd
from duke_dbt_data import dcmread_image

df = pd.read_csv("data/BCS-DBT file-paths-train-v2.csv")

view_series = df.iloc[0]
view = view_series["View"]

image_path = os.path.join("D:/Licenta/Proiect/saveData/trainData\manifest-1617905855234/", view_series["descriptive_path"])

image = dcmread_image(fp=image_path, view=view)

plt.imshow(image[0], cmap=plt.cm.gray)
plt.show()