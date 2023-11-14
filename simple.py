import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import hog
from skimage.measure import regionprops
from scipy import stats
from skimage.measure import label
import joblib

class PlantDiseaseApp:
    def __init__(self, root, pca_model, scaler, apple_model, corn_model, grapes_model, potato_model, tomato_model):
        self.root = root
        self.root.title("Plant Disease Prediction")

        self.selected_plant_type = tk.StringVar()
        self.selected_plant_type.set('Apple')

        self.pca_model = pca_model
        self.scaler = scaler
        self.apple_model = apple_model
        self.corn_model = corn_model
        self.grapes_model = grapes_model
        self.potato_model = potato_model
        self.tomato_model = tomato_model

        self.create_widgets()

    def create_widgets(self):
        # Choose Plant Type
        plant_type_label = tk.Label(self.root, text="Select Plant Type:")
        plant_type_label.pack(pady=10)

        plant_types = ['Apple', 'Corn', 'Grapes', 'Potato', 'Tomato']
        plant_type_menu = tk.OptionMenu(self.root, self.selected_plant_type, *plant_types)
        plant_type_menu.pack(pady=10)

        # Upload Image
        upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        upload_button.pack(pady=10)

        # Predict Button
        predict_button = tk.Button(self.root, text="Predict", command=self.predict_disease)
        predict_button.pack(pady=10)

        # Image Display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Disease Prediction Result
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.image_path = file_path

            # Display the selected image
            img = Image.open(file_path)
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img

    def preprocess_image(self):
        names = ['area', 'perimeter', 'physiological_length', 'physiological_width', 'aspect_ratio',
             'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b',
             'contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity',
             'eccentricity', 'solidity', 'equiv_diameter', 'major_axis_length', 'minor_axis_length',
             'skewness', 'kurtosis', 'hog_feature_1', 'hog_feature_2', 'hog_feature_3'
            ]
        df = pd.DataFrame(columns=names)
    
        # Load the image and perform preprocessing
        img = Image.open(self.image_path)
        main_img = np.array(img)

        # Preprocessing
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25, 25), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((50, 50), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Shape features
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Color features
        red_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        blue_channel = img[:, :, 2]

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        # Std deviation
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        # Texture features using GLCM matrix
        glcm = graycomatrix(gs,
                            distances=[1],
                            angles=[0],
                            symmetric=True,
                            normed=True)

        properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']
        contrast = graycoprops(glcm, properties[0])
        energy = graycoprops(glcm, properties[1])
        homogeneity = graycoprops(glcm, properties[2])
        correlation = graycoprops(glcm, properties[3])
        dissimilarity = graycoprops(glcm, properties[4])

        # Statistical moments
        skewness = stats.skew(gs.flatten())
        kurtosis = stats.kurtosis(gs.flatten())

        # Additional features
        labeled_img = label(closing)
        regions = regionprops(labeled_img)

        eccentricity = regions[0].eccentricity
        solidity = regions[0].solidity
        equiv_diameter = regions[0].equivalent_diameter
        major_axis_length = regions[0].major_axis_length
        minor_axis_length = regions[0].minor_axis_length

        # HOG features
        hog_features = hog(gs, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys')

        vector = [area, perimeter, w, h, aspect_ratio,
                        red_mean, green_mean, blue_mean, red_std, green_std, blue_std,
                        contrast[0][0], energy[0][0], homogeneity[0][0], correlation[0][0], dissimilarity[0][0],
                        eccentricity, solidity, equiv_diameter, major_axis_length, minor_axis_length,
                        skewness, kurtosis, hog_features[0], hog_features[1], hog_features[2]]

        df_temp = pd.DataFrame([vector], columns=names)
        df = pd.concat([df, df_temp], ignore_index=True)

        return df

    def predict_disease(self):
        if hasattr(self, 'image_path'):
            # Preprocess the image for model prediction
            processed_image = self.preprocess_image()

            # Apply PCA transformation
            pca_features = self.pca_model.transform(processed_image)

            # Normalize using the loaded scaler
            normalized_data = self.scaler.transform(pca_features)

            # Predict using the classifier model
            if self.selected_plant_type.get() == 'Apple':
                classifier_model = self.apple_model
            elif self.selected_plant_type.get() == 'Corn':
                classifier_model = self.corn_model
            elif self.selected_plant_type.get() == 'Grapes':
                classifier_model = self.grapes_model
            elif self.selected_plant_type.get() == 'Potato':
                classifier_model = self.potato_model
            elif self.selected_plant_type.get() == 'Tomato':
                classifier_model = self.tomato_model

            prediction = classifier_model.predict(normalized_data)

            # Map the prediction to the actual disease type (modify as needed)
            disease_type = prediction

            # Display the result
            self.result_label.config(text=f"Predicted Disease: {disease_type}")

# Load PCA model, StandardScaler, and classifier models
pca_model = joblib.load('models/pca_model.pkl')
scaler = joblib.load('models/scaler_model.pkl')
apple_model = joblib.load('models/classifiers/Apple_Classifier_model.pkl')
corn_model = joblib.load('models/classifiers/classifier_model.pkl')
grapes_model = joblib.load('models/classifiers/Grapes_Classifier_model.pkl')
potato_model = joblib.load('models/classifiers/Potato_Classifier_model.pkl')
tomato_model = joblib.load('models/classifiers/Tomato_Classifier_model.pkl')

if __name__ == '__main__':
    root = tk.Tk()
    app = PlantDiseaseApp(root, pca_model, scaler, apple_model, corn_model, grapes_model, potato_model, tomato_model)
    root.mainloop()
