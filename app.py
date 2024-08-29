from flask import Flask, redirect, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Cargar el modelo entrenado
model_path = 'modelos/modelo_entrenado.h5'  # Ajusta la ruta al modelo entrenado
model = tf.keras.models.load_model(model_path)

# Lista de etiquetas de enfermedades
disease_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", 
                  "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding", 
                  "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((148, 148))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar la imagen
    return img_array

@app.route('/img')
def img():
    return render_template('upload_img.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    img_data = file.read()
    img_array = preprocess_image(img_data)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Obtener el nombre de la enfermedad predicha
    predicted_disease = disease_labels[predicted_class]

    return render_template('result.html', prediction=predicted_disease)

if __name__ == '__main__':
    app.run(debug=True)
