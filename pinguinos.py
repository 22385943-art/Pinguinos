from flask import Flask, render_template, request, jsonify
import cohere
import json
import pickle
import onnxruntime
import numpy as np
import os
import datetime
import random
from pymongo import MongoClient
#Leemos el.env
from dotenv import load_dotenv
load_dotenv() #se le pone la ruta al .env

#Leer la variable de entorno cohere_Api_key
import os
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=cohere_api_key)

try:
    mongo_client = MongoClient(os.getenv("MONGO_URI"))
    db = mongo_client["penguin_app"]
    collection = db["predictions"]
    print("MongoDB Conectado.")
except Exception as e:
    print(f"Error")

img_url = "https://thumbs.dreamstime.com/b/ping%C3%BCino-21267520.jpg?w=768"

try:
    onnxruntime_session = onnxruntime.InferenceSession("penguins_rf.onnx")
except FileNotFoundError:
    read_rf = None

app = Flask(__name__)

NICKNAMES = [
    "Kowalski", "Capitán Frío", "Happy Feet", "Sir Waddles", "Pingu", 
    "Iceberg", "Snowball", "El Padrino", "Sargento", "Frosty",
    "Mr. Smoking", "Pescador", "Tornado", "Emperador", "Comandante", 
    "Glacier", "Flipper", "Yeti", "Pebble", "Fishlover", "Gentleman", "Torpedo"
]

HABITATS = {
    "ADELIE":   {"center": [-77.0, 166.0], "spread": 10.0}, # Mar de Ross / General
    "CHINSTRAP": {"center": [-62.0, -58.0], "spread": 2.0}, # Islas Shetland del Sur
    "GENTOO":   {"center": [-52.0, -59.0], "spread": 3.0},  # Malvinas/Falklands
    "UNKNOWN":  {"center": [-70.0, 0.0],   "spread": 20.0}
}

def get_random_coords(species):
    """Genera una coordenada aleatoria realista basada en la especie."""
    key = species.upper()
    if key not in HABITATS:
        key = "UNKNOWN"
    
    habitat = HABITATS[key]
    # Generamos un desplazamiento aleatorio alrededor del centro
    lat = habitat["center"][0] + random.uniform(-habitat["spread"], habitat["spread"])
    lon = habitat["center"][1] + random.uniform(-habitat["spread"], habitat["spread"])
    return lat, lon

def get_features_from_image(img_url):
    # Prompt Ingeniería Inversa: Ayudamos al LLM a no alucinar medidas imposibles
    prompt = """
    Eres un asistente de visión artificial. Tu trabajo NO es estimar medidas, sino CLASIFICAR visualmente y asignar el perfil biométrico correcto.

    PASO 1: Analiza la imagen y decide la especie basándote en rasgos visuales:
    
    A) ADELIE (Adelia): 
       - Rasgo clave: Anillo blanco visible alrededor del ojo.
       - Pico: Corto y mayormente negro/oscuro.
       - Cabeza: Totalmente negra.
       
    B) CHINSTRAP (Barbijo):
       - Rasgo clave: Línea negra fina debajo de la barbilla (como un casco).
       - Cara: Blanca hasta encima de los ojos.
       
    C) GENTOO (Papúa):
       - Rasgo clave: Mancha blanca triangular sobre los ojos (como una diadema).
       - Pico: Naranja brillante o rojo.
       - Patas: Muy naranjas.

    PASO 2: Asigna los valores EXACTOS para la especie detectada:

    SI ES ADELIE -> Usa estos valores (Pingüino pequeño):
    { "bill_length_mm": 38.0, "bill_depth_mm": 18.0, "flipper_length_mm": 185.0, "body_mass_g": 3400.0, "sex": 1 }

    SI ES CHINSTRAP -> Usa estos valores (Pingüino mediano):
    { "bill_length_mm": 49.0, "bill_depth_mm": 19.0, "flipper_length_mm": 195.0, "body_mass_g": 3800.0, "sex": 1 }

    SI ES GENTOO -> Usa estos valores (Pingüino gigante):
    { "bill_length_mm": 48.0, "bill_depth_mm": 14.5, "flipper_length_mm": 220.0, "body_mass_g": 5200.0, "sex": 1 }

    INSTRUCCIÓN FINAL:
    Si dudas entre Adelie y Chinstrap, mira el ojo. Si tiene anillo blanco, es ADELIE.
    Devuelve SOLO el JSON correspondiente a la especie que ves.
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

@app.route("/api/community", methods=['GET'])
def get_community_penguins():
    if collection is None:
        return jsonify([])
    
    # Obtenemos los últimos 50 pingüinos (excluyendo el campo _id que da problemas con JSON)
    penguins = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
    return jsonify(penguins)

@app.route("/inicio", methods = ["POST", "GET"])
def inicio():
    prediction_text = "No se ha realizado predicción"
    img_url = None
    features_dict = None
    
    # Variables por defecto para que no falle si es GET
    nickname = "Desconocido"
    lat, lon = -90, 0
    resultado_especie = "UNKNOWN"

    if request.method == "POST":
        img_url = request.form.get("img_url")
        features_dict = get_features_from_image(img_url)
        
        bill_len = features_dict["bill_length_mm"]
        bill_dep = features_dict["bill_depth_mm"]
        flipper = features_dict["flipper_length_mm"]
        mass = features_dict["body_mass_g"]
        sex = features_dict["sex"]

        input_data = np.array(
            [[bill_len, bill_dep, flipper, mass, sex]], 
            dtype=np.float32
        )

        if onnxruntime_session:
            input_name = "features" 
            outputs = onnxruntime_session.run(None, {input_name: input_data})
            pred_class = outputs[0][0]
            
            species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
            resultado_especie = species_map.get(pred_class, "UNKNOWN")
            
            prediction_text = (f"Para un pingüino con pico de {bill_len}mm, {bill_dep}mm, "
                               f"aleta de {flipper}mm, peso de {mass}g, y de sexo {sex}, "
                               f"el modelo predice que es de la especie: {resultado_especie.upper()}")
                        
            # A. Generar Nickname y Coordenadas
            nickname = random.choice(NICKNAMES)
            lat, lon = get_random_coords(resultado_especie)

            if collection is not None:
                try:
                    doc = {
                        "timestamp": datetime.datetime.utcnow(),
                        "img_url": img_url,
                        "features": features_dict,
                        "species": resultado_especie.upper(), # Guardar en mayúsculas
                        "nickname": nickname,
                        "coords": {"lat": lat, "lon": lon} # Objeto coords para el mapa
                    }
                    collection.insert_one(doc)
                    print("Pingüino registrado en la comunidad.")
                except Exception as e:
                    print(f"Error guardando en Mongo: {e}")

        else:
            prediction_text = "Error:El modelo no está cargado."

    # Pasamos las nuevas variables al HTML
    return render_template("frontendresultado.html", 
                           nombre="Usuario", 
                           poema=prediction_text, 
                           img_url=img_url,
                           features=features_dict,
                           nickname=nickname,
                           my_lat=lat,
                           my_lon=lon,
                           species=resultado_especie.upper())

@app.route("/navidad", methods=["POST"])
def navidad():
    # Recogemos los datos que nos envía el frontend principal
    nickname = request.form.get("nickname")
    species = request.form.get("species")
    img_url = request.form.get("img_url")
    
    # Renderizamos la nueva página festiva
    return render_template("navidad.html", 
                           nickname=nickname, 
                           species=species, 
                           img_url=img_url)

@app.route("/presentacion", methods=['GET'])
def ver_presentacion():
    return render_template("presentacion.html")


if __name__ == "__main__":  
    app.run(debug=True, host="localhost", port=5000)