from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app= Flask(__name__)
car=pd.read_csv("final car.csv")
model= pickle.load(open("linreg_pickle.pkl","rb"))

@app.route('/')
def index():
    car_model= sorted(car["Car_Name"].unique(),reverse=True)
    present_price= sorted(car["Present_Price"].unique(),reverse=True)
    car_age= sorted(car["car_age"].unique(),reverse=True)
    km_driven = sorted(car["Kms_Driven"].unique(),reverse=True)
    fuel_type= car["Fuel_Type"].unique()
    transmission = car["Transmission"].unique()
    seller_type = car["Seller_Type"].unique()
    car_model.insert(0, 'Select Company')
    return render_template('index.html', car_model=car_model,present_price=present_price,car_age=car_age,km_driven=km_driven,
fuel_type=fuel_type , transmission= transmission , seller_type = seller_type )



@app.route('/predict',methods=['POST'])
def predict():
    Model=request.form.get('Model')
    age=request.form.get('age')
    selller=request.form.get('selller')
    fuel=request.form.get('fuel')
    trans=request.form.get('trans')
    presentprice=request.form.get('presentprice')
    kilo_driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['Car_Name','car_age','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission'],
                                          data=np.array(['Model','age', 'presentprice', 'kilo_driven','fuel','selller','trans']).reshape(1,7)))


    print(prediction)

    return str(np.round(prediction[0],2))

if  __name__ == "__main__":

    app.run(debug=True)