from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
lr=pickle.load(open('co2.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/rag')
def rag():
    return render_template("co2.html")

@app.route('/predict',methods=['post'])
def predict():
    Modelyear=float(request.form['MODELYEAR'])
    Make=float(request.form['MAKE'])
    Model=float(request.form['MODEL'])
    VehicleClass=float(request.form['VEHICLECLASS'])
    EngineSize=float(request.form['ENGINESIZE'])
    Cylinders=float(request.form['CYLINDERS'])
    Transmission=float(request.form['TRANSMISSION'])
    FuelType=float(request.form['FUELTYPE'])
    FuelConsumptionCity=float(request.form['FUELCONSUMPTIONCITY'])
    FuelConsumptionHwy=float(request.form['FUELCONSUMPTIONHWY'])
    FuelConsumptionComb=float(request.form['FUELCONSUMPTIONCOMB'])
    FuelConsumptionCombMpg=float(request.form['FUELCONSUMPTIONCOMBMPG'])

    a=np.array([[Modelyear,Make,Model,VehicleClass,EngineSize,Cylinders,Transmission,FuelType,
    FuelConsumptionCity,FuelConsumptionHwy,FuelConsumptionComb,FuelConsumptionCombMpg]])
    print(a)

    result=lr.predict(a)
    
    x=result

    return render_template('co2.html',x='Value is : {}'.format(*x))

if __name__ == '__main__':
    app.run()