from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.models import model_from_json


PROJECT_DIR = Path(__file__).resolve().parent
EMOTIONS_LIST = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


def FacialExpressionModel(json_file: Path, weights_file: Path):
    with json_file.open("r", encoding="utf-8") as file:
        model = model_from_json(file.read())

    model.load_weights(str(weights_file))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


top = tk.Tk()
top.geometry("800x600")
top.title("Emotion Detector")
top.configure(background="#CDCDCD")

label1 = tk.Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
sign_image = tk.Label(top)

facec = cv2.CascadeClassifier(
    str(PROJECT_DIR / "haarcascade_frontalface_default.xml")
)
if facec.empty():
    raise RuntimeError("Could not load haarcascade_frontalface_default.xml")

model = FacialExpressionModel(
    PROJECT_DIR / "model_a1.json",
    PROJECT_DIR / "model_weights1.h5",
)


def Detect(file_path: Path):
    image = cv2.imread(str(file_path))
    if image is None:
        label1.configure(foreground="#011638", text="Unable to read image")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    if len(faces) == 0:
        label1.configure(foreground="#011638", text="No face detected")
        return

    for x, y, width, height in faces:
        face = gray_image[y : y + height, x : x + width]
        roi = cv2.resize(face, (48, 48))
        prediction = model.predict(roi[np.newaxis, :, :, np.newaxis], verbose=0)
        emotion = EMOTIONS_LIST[int(np.argmax(prediction[0]))]

    print(f"Predicted Emotion is {emotion}")
    label1.configure(foreground="#011638", text=emotion)


def show_Detect_button(file_path: Path):
    detect_button = tk.Button(
        top,
        text="Detect Emotion",
        command=lambda: Detect(file_path),
        padx=10,
        pady=5,
    )
    detect_button.configure(
        background="#364156",
        foreground="white",
        font=("arial", 10, "bold"),
    )
    detect_button.place(relx=0.79, rely=0.46)


def upload_image():
    selected_path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All files", "*.*"),
        ]
    )
    if not selected_path:
        return

    image_path = Path(selected_path)
    try:
        with Image.open(image_path) as uploaded:
            uploaded.thumbnail((top.winfo_width() / 2.25, top.winfo_height() / 2.25))
            image_preview = ImageTk.PhotoImage(uploaded.copy())
    except (OSError, tk.TclError) as error:
        messagebox.showerror("Unable to open image", str(error), parent=top)
        return

    sign_image.configure(image=image_preview)
    sign_image.image = image_preview
    label1.configure(text="")
    show_Detect_button(image_path)


upload = tk.Button(
    top,
    text="Upload Image",
    command=upload_image,
    padx=10,
    pady=5,
)
upload.configure(
    background="#364156",
    foreground="white",
    font=("arial", 20, "bold"),
)
upload.pack(side="bottom", pady=50)
sign_image.pack(side="bottom", expand=True)
label1.pack(side="bottom", expand=True)

heading = tk.Label(top, text="Emotion Detector", pady=20, font=("arial", 25, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
