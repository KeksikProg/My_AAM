import os
import cv2
import pandas as pd

def read_dataset(image_dir, csvfile_path):
	image_names = os.listdir(image_dir)
	landmarks_df = pd.read_csv(csvfile_path)

	landmark_columns = landmarks_df.columns.tolist()[2:]
	image_list, landmarks_list = [], []

	for img_name in image_names:
		image_path = os.path.join(image_dir, img_name)
		image_list.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

		landmarks_df_line = landmarks_df[landmarks_df["name"] == img_name.replace(".jpg", "")]
		landmarks = []


		for i in range(0, len(landmark_columns), 2):
			x = landmarks_df_line[landmark_columns[i]].iloc[0]
			y = landmarks_df_line[landmark_columns[i+1]].iloc[0]			
			landmarks.append((x, y))
		landmarks_list.append(landmarks)
	return image_list, landmarks_list