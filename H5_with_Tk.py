# IMPORTS
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox


# Get model
model = load_model('prediction_model.h5')


# Prompt user to select a file
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
print(file_path)  # GET FILE PATH


# Preprocess image to feed the model
img = Image.open(file_path).convert('L').resize((48, 48))
ITK = ImageTk.PhotoImage(img)
np_img = np.array(img)
img_array = np_img.reshape(48, 48, 1)


# Use model to predict

# angry: 0, happy: 1, sad: 2
hold_pred = np.argmax(model.predict(img_array[None,:,:]))

def prompt(message, img):
    messagebox.showinfo('Predictions', message)
    

if hold_pred == 0:
    prompt('Angry', ITK)
elif hold_pred == 1:
    prompt('Happy', ITK)
elif hold_pred == 2:
    prompt('Sad', ITK)