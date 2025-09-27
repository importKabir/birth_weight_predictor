from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])
    
    cleaned_data = {"gestation":[gestation],
                    "parity":[parity],
                    "age":[age],
                    "height":[height],
                    "weight":[weight],
                    "smoke":[smoke]
                    }
    
    return cleaned_data

@app.route("/", methods = ['GET'])
def home():
    return render_template("index.html")


#DEFINE ENDPOINTS
@app.route("/predict", methods = ['POST'])
def get_prediction():
    # get data from user
    baby_data_form = request.form.to_dict(flat=True)
    baby_data_form['smoke'] = 1 if baby_data_form.get('smoke') else 0
    
    baby_data_cleaned = get_cleaned_data(baby_data_form)


    #convert it into dataframe
    baby_df = pd.DataFrame(baby_data_cleaned)

    #load trained ML model 
    with open("model/model.pkl", 'rb') as obj:
        model = pickle.load(obj) 

    #make predictions on user data
    prediction = round(float(model.predict(baby_df)), 2)   

    #return response in json format
    response = {"Prediction":prediction}
    
    return render_template("index.html", prediction=prediction)




if __name__ == '__main__':
    app.run(debug=True)
