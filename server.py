from flask import Flask, request, jsonify, render_template, session
import numpy as np
import pandas as pd
from joblib import load, dump
import os
from flask_cors import CORS

#paths
upload_folder_path = os.path.join('static', 'uploads')
#Cargar el modelo
dt = load('model.joblib')

#Generar el servidor
servidorWeb = Flask(__name__, template_folder='templates', static_folder='static')
servidorWeb.config['UPLOAD_FOLDER'] = upload_folder_path
CORS(servidorWeb)

@servidorWeb.route('/modelo', methods=['GET'])
def form():
    return render_template('form1.html')

@servidorWeb.route('/modelo/reentrenar', methods=['POST', 'GET'])
def reentrenar():
    if request.method == "POST":
        contenido = request.json
        lista = contenido['DB']
        print(lista)
        dataFrame = pd.DataFrame.from_records(lista)

        dataFrame['resultado'] = dataFrame['resultado'].replace([0,1],['noDiabetico', 'Diabetico'])

        #Caracteristicas de entrada (Informacion de los campos del formulario)
        X = dataFrame.drop('resultado', axis=1)
        #Caracteristicas de salida ()
        y = dataFrame['resultado']

        #Separar la bas de datos en dos conjuntos entrenamiento (guia de estudio) y prueba(examen)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

        from sklearn import svm
        kernels = ['linear', 'poly', 'rbf']
        models = []

        for ker in kernels:
            dt = svm.SVC(kernel=ker)
            dt.fit(X_train, y_train)
            models.append(dt)

        bestModel = None
        bestScore = 0
        for model in models:
            actualScore = model.score(X_test, y_test)  
            if actualScore >= bestScore:
                bestScore = actualScore
                bestModel = model

        dump(bestModel, 'model.joblib')
        return f"Reentrenado correctamente, score: {bestScore}"



@servidorWeb.route('/modelo/prediccion', methods=['POST', 'GET'])
def model():
  if request.method == "POST":
    #Procesar datos de la entrada
    #contenido = request.json
    contenido = request.json
    '''
    if request.files['file']:
        file = request.files['file']
        file_name = secure_filename(file.filename)
        file.save(os.path.join(servidorWeb.config['UPLOAD_FOLDER'], file_name))
        data_file = pd.read_csv(os.path.join(servidorWeb.config['UPLOAD_FOLDER'], file_name))
        data_file_list = data_file.to_numpy()
        for row in data_file_list:
            res = dt.predict(row.reshape(1, -1))
            lista = row.tolist()
            lista.append(res[0])
            table.append(lista)

    '''
    datosEntrada = np.array([
        contenido['embarazos'],
        contenido['glucosa'],
        contenido['presion'],
        contenido['grosorPiel'],
        contenido['insulina'],
        contenido['bmi'],
        contenido['dpf'],
        contenido['edad'],
        ])
    #utilizar el modelo 
    resultado = dt.predict(datosEntrada.reshape(1, -1))
    #Regresar la salida del modelo
    #return jsonify({"Resultado":str(resultado[0])})
    resultado = resultado.tolist()
    return resultado[0]
    #return render_template('Table1.html', table=table)

if __name__ == '__main__':
    servidorWeb.run(debug=True,host='0.0.0.0',port='8081')