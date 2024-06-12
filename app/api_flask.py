from flask import Flask, request, jsonify
from flask import Flask
#from flask_swagger_ui import get_swaggerui_blueprint
from flask_restx import Api, Resource, reqparse
#import joblib
#import pandas as pd 
#import numpy as np

from rf_model import Modelo_LSTM

app = Flask(__name__)
model = Modelo_LSTM()

api = Api(app, version='1.0',
           title='ICD API',
           description='API for ICD')

ns = api.namespace('icd lstm', description='Clasificación ICD con LSTM')

parser = reqparse.RequestParser()
parser.add_argument('texto', type = str, help = 'Texto a clasificar')

@ns.route('/hello')
class HelloWorld(Resource):
    def get(self):

        return {'hello': 'API para la codificación de un texto libre en clave ICD'}


@ns.route('/predict',)
class Predict(Resource):
    @api.doc(parser = parser)
    def get(self):
        texto = parser.parse_args()
        #print(texto['texto'])
        causa = int(model.prediction(texto['texto']))
        #print(causa)
        return jsonify({'texto':texto['texto'], 
                        'causa':causa})
# @api.route('/get_cie')
# class HelloWorld(Resource):
#     def get_cie(self):
#         if request.is_json:
#             # Get JSON data from request
#             features = 'diabetes'#request.get_json()

#             # Convert features to DataFrame
#             features_df = pd.DataFrame([features])
            
#             # Apply preprocessing
#             processed_features = preprocessor.transform(features_df)

#             # Ensure the input is shaped correctly for a single sample
#             if processed_features.ndim == 1:
#                 processed_features = processed_features.reshape(1, -1)

#             # Make prediction
#             prediction = model.predict(processed_features)
            
#             # Check if the output is already a scalar and convert it to Python native type for JSON serialization
#             if isinstance(prediction, np.ndarray):
#                 prediction = prediction.item()  # Extract single numpy value as Python scalar
#             else:
#                 prediction = int(prediction)  # Convert numpy int64 to Python native int
#             # Return the prediction in JSON format
#             return jsonify({"predicted class": prediction[0]}), 200
#         else:
#             return jsonify({"error": "Request must be in JSON format"}), 400
        
    # # Asegúrate de que estás recibiendo un JSON
    # if request.is_json:
    #     # Obtiene los datos en formato JSON
    #     data = request.get_json()
        
    #     # Aquí puedes procesar tus datos
    #     print(data)  # Imprime los datos, puedes hacer algo más útil
        
    #     # Envía una respuesta
    #     response = {"message": "Datos recibidos", "tusDatos": data}
    #     return jsonify(response), 200
    # else:
    #     return jsonify({"error": "Request must be JSON"}), 400

   

if __name__ == '__main__':
    app.run(debug=True)

