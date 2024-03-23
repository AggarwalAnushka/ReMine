from flask import Flask, render_template, request
# from models import get_mining_technique
from models import *
from models import check
import pandas as pd
import os

app = Flask(__name__,static_folder='static')

def load_dataset():
    # Get the path to the CSV file
    data_folder = os.path.join(os.getcwd(), 'data')
    csv_path = os.path.join(data_folder, 'Dataset1.csv')

    # Load the dataset using Pandas
    df = pd.read_csv(csv_path)

    # You can perform any additional processing or checks here if needed

    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<state_name>', methods=['GET','POST'])
def predict(state_name):
    # # Get data from the frontend
    # data = request.form['input_data']

    # # Use your AI model function
    # result = get_mining_technique(state_name)
    # print(state_name)
    # result= check(state_name)
    df= load_dataset()
    result= predict_resource_and_mining(state_name,df)

    # Pass the result to the template
    # return render_template('model.html', result=result)
    print(result)
    return render_template('table.html',result=result, state_name=state_name)

@app.route('/region', methods=['GET','POST'])
def region():
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True)