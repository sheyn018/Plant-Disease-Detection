from flask import Flask, render_template, request, redirect, url_for
import joblib
from PIL import Image
import numpy as np
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import hog
from skimage.measure import regionprops
from scipy import stats
from skimage.measure import label
import joblib

app = Flask(__name__)

# Load PCA model
with open('pca_model.pkl', 'rb') as file:
    pca_model = joblib.load(file)

# Load StandardScaler
with open('scaler_model.pkl', 'rb') as file:
    scaler = joblib.load(file)

# Load classifier model
with open('classifier_model.pkl', 'rb') as file:
    classifier_model = joblib.load(file)

# Define a temporary directory to store uploaded files
UPLOAD_FOLDER = 'static'  # Use 'static' for simplicity
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    # Features
    names = ['area', 'perimeter', 'physiological_length', 'physiological_width', 'aspect_ratio',
             'mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b',
             'contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity',
             'eccentricity', 'solidity', 'equiv_diameter', 'major_axis_length', 'minor_axis_length',
             'skewness', 'kurtosis', 'hog_feature_1', 'hog_feature_2', 'hog_feature_3'
            ]
    df = pd.DataFrame(columns=names)

    # Load the image and perform preprocessing
    img = Image.open(image_path)
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

def predict_disease(image_array):
    # Apply PCA transformation
    pca_features = pca_model.transform(image_array)

    # Normalize using the loaded scaler
    normalized_data = scaler.transform(pca_features)

    # Predict using the classifier model
    prediction = classifier_model.predict(normalized_data)

    # Map the prediction to the actual disease type (modify as needed)
    disease_type = prediction

    return disease_type

@app.route('/', methods=['GET', 'POST'])
def index():
    plant_type = ''
    disease_type = ''
    
    if request.method == 'POST':
        # Get user input
        uploaded_file = request.files['photo']
        selected_plant_type = 'Pepper'

        # Save the uploaded photo temporarily
        photo_path = 'static/temp.jpg'  # Use 'static' for simplicity
        uploaded_file.save(photo_path)

        # Preprocess the image for model prediction
        processed_image = preprocess_image(photo_path)

        # Predict the disease type
        disease_type = predict_disease(processed_image)

    return render_template('index.html', plant_type='Pepper', disease_type=disease_type)

if __name__ == '__main__':
    app.run(debug=True)
