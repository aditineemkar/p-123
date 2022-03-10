import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get("PYTHONHTTPSVERIFY") and getattr(ssl, "_create_unverified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
#print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

samples_per_class = 5
fig = plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=3750, test_size=1250)
X_train_scaled = X_train / 255.0    
X_test_scaled = X_test / 255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print(accuracy_score(y_test, y_pred))


cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    try:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        lower_right = (int(width / 2 + 56), int(height / 2 + 56))

        cv2.rectangle(gray, upper_left, lower_right, (0, 255, 0), 2)

        roi = gray[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]


        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert("L")
        image_bw_resized = image_bw.resize((22,30), Image.ANTIALIAS)

        image_bw_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20

        min_pixel = np.percentile(image_bw_inverted, pixel_filter)
        image_bw_scaled = np.clip(image_bw_inverted - min_pixel, 0, 255)
        max_pixel = np.max(image_bw_inverted)
        image_bw_scaled = np.asarray(image_bw_scaled)/max_pixel

        test_sample = np.array(image_bw_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)

        cv2.imshow("frame", gray)

    except:
        print("ERROR")

cap.release()
cv2.destroyAllWindows()
