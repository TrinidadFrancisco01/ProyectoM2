from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo
model = joblib.load('./random_forest_model.pkl')
app.logger.debug("Modelo cargado exitosamente.")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        co2 = float(request.form['CO2(tCO2)'])
        reactive = float(request.form['Lagging_Current_Reactive.Power_kVarh'])
        lagging = float(request.form['Lagging_Current_Power_Factor'])
        leading = float(request.form['Leading_Current_Power_Factor'])
        nsm = float(request.form['NSM'])
        load_type = int(request.form['Load_Type'])  # valor ya codificado (0,1,2)

        # Crear DataFrame
        data_df = pd.DataFrame([[co2, reactive, lagging, leading, nsm, load_type]],
            columns=['CO2(tCO2)', 'Lagging_Current_Reactive.Power_kVarh', 
                     'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 
                     'NSM', 'Load_Type'])

        app.logger.debug(f"Datos recibidos: \n{data_df}")

        # Realizar predicción
        prediction = model.predict(data_df)
        app.logger.debug(f"Predicción realizada: {prediction[0]}")

        return jsonify({'Usage_kWh': round(float(prediction[0]), 2)})

    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
