import cv2
import pandas as pd
import numpy as np

target_size = (200, 200)
path_prefix = '/Users/luizfrf/Documents/mo809/Ch2_001/'
center_folder = 'center/'
labels_file = 'final_example.csv'
file_extension = '.jpg'

def load_img(img_name):
    img = cv2.imread(path_prefix + center_folder + img_name + file_extension)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = (img / 127.5) - 1.0
    return [img]

df = pd.read_csv(path_prefix + labels_file)
df.frame_id = df.frame_id.astype(str)
imgs = []
ground_truths = []
for i, row in df.head(100).iterrows():
    imgs.append(load_img(row['frame_id']))
    ground_truths.append(row['steering_angle'])

imgs = np.array(imgs, dtype=np.float32)
ground_truths = np.array(ground_truths, dtype=np.float32)

np.savez('data/data.npz', imgs=imgs, ground_truths=ground_truths)
