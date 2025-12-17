from flask import Flask, render_template, request
import cohere
import json
import pickle
import onnxruntime
import numpy as np
import os
import datetime
from pymongo import MongoClient
#Leemos el.env
from dotenv import load_dotenv
load_dotenv() #se le pone la ruta al .env

#Leer la variable de entorno cohere_Api_key
import os
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=cohere_api_key)

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["penguin_app"]
collection = db["predictions"]

img_url = "https://thumbs.dreamstime.com/b/ping%C3%BCino-21267520.jpg?w=768"

try:
    onnxruntime_session = onnxruntime.InferenceSession("penguins_rf.onnx")
except FileNotFoundError:
    read_rf = None

app = Flask(__name__)


def get_features_from_image(img_url):
    # Definimos el prompt con instrucciones estrictas (System Prompt)
    prompt = """
    Eres un ornitólogo experto y un asistente de datos preciso. Recibes una imagen de un pingüino y tu ÚNICA tarea es devolver SIEMPRE un JSON con 5 campos numéricos.

    INSTRUCCIONES IMPORTANTES:
    
    1. FORMATO DE SALIDA
    - Debes responder SIEMPRE y SOLO con un objeto JSON válido.
    - No incluyas texto antes ni después (ni "Aquí tienes", ni bloques markdown ```json).
    - El formato exacto debe ser:
    {
        "bill_length_mm": 45.5,
        "bill_depth_mm": 14.2,
        "flipper_length_mm": 210.0,
        "body_mass_g": 4200.0,
        "sex": 1
    }

    2. SIGNIFICADO DE LOS CAMPOS (Estimaciones Biométricas)
    - "bill_length_mm" (float): Longitud del pico en mm. Rango típico: 30.0 a 60.0.
    - "bill_depth_mm" (float): Profundidad del pico en mm. Rango típico: 13.0 a 22.0.
    - "flipper_length_mm" (float): Longitud de la aleta en mm. Rango típico: 170.0 a 235.0.
    - "body_mass_g" (float): Masa corporal en gramos. Rango típico: 2700.0 a 6500.0.
    - "sex" (int): Sexo estimado del pingüino.
        * 0 = Hembra
        * 1 = Macho

    3. REGLAS DE ESTIMACIÓN
    - Si la imagen es un dibujo, caricatura o no es clara, INVÉNTATE valores realistas dentro de los rangos típicos.
    - El modelo ONNX necesita números, así que NUNCA devuelvas null.
    - Usa tu mejor criterio visual para estimar si es un pingüino grande (Gentoo) o pequeño (Adelie) y ajusta el peso y aletas acorde.

    4. REGLA DE ORO
    - PASE LO QUE PASE devuelve JSON.
    """

    response = co.chat(
        model="command-a-vision-07-2025",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                            "detail": "auto"
                        }
                    }
                ]
            }
        ]
    )

    return json.loads(response.message.content[0].text)

@app.route("/", methods = ['GET'])
def home():
    return render_template("frontend.html")

@app.route("/inicio", methods = ["POST", "GET"])
def inicio():
    prediction = None
    if request.method == "POST":
        img_url = request.form.get("img_url")
        features_dict = get_features_from_image(img_url)
        bill_len = features_dict["bill_length_mm"]
        bill_dep = features_dict["bill_depth_mm"]
        flipper = features_dict["flipper_length_mm"]
        mass = features_dict["body_mass_g"]
        sex = features_dict["sex"]

        #Convertimos las características en el array en el orden correcto y con variables float, tal y como espera el modeo 
        input_data = np.array(
            [[bill_len, bill_dep, flipper, mass, sex]], 
            dtype=np.float32
        )

        #Hacemos la predicción
        if onnxruntime_session:
            input_name = "features" 
            outputs = onnxruntime_session.run(None, {input_name: input_data})
            
            pred_class = outputs[0][0]
            
            species = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
            resultado_especie = species.get(pred_class, "Desconocido")
            prediction = (f"Para un pingüino con pico de {bill_len}mm, {bill_dep}mm,aleta de {flipper}mm, peso de {mass}g, y de sexo {sex},el modelo predice que es de la especie:{resultado_especie.upper()}")
            
            # Aquí tenemos que insertar en Mongo
            #...
        else:
            prediction = "Error:El modelo no está cargado"

    return render_template("frontendresultado.html", 
                           nombre="Usuario", 
                           poema=prediction, 
                           img_url=img_url,
                           features=features_dict) # Pasamos las features por si quieres mostrarlas

if __name__ == "__main__":  
    app.run(debug=True, host="localhost", port=5000)