from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open('model_and_encoders.pkl', 'rb') as model_file:
    pickle_file = pickle.load(model_file)

model = pickle_file['model']
label_encoders = pickle_file['label_encoders']
scaler = pickle_file['scale']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    restaurant_type = request.form['restaurant_type']
    rate = request.form['rate']
    ratings = request.form['ratings']
    online_order = request.form['online_order']
    table_booking = request.form['table_booking']
    cuisines_type = request.form['cuisines_type']
    area = request.form['area']

    new_data = [[restaurant_type, rate, ratings, online_order, table_booking, cuisines_type, area]]
    
    columns = ['restaurant type', 'rate (out of 5)', 'num of ratings', 'online_order', 'table booking', 'cuisines type','area']

    # Define data types for specific columns using a dictionary
    dtype_dict = {'rate (out of 5)': float, 'num of ratings': int}

    # Create the DataFrame with specified data types
    new_df = pd.DataFrame(new_data, columns=columns)

    # Convert specific columns to the specified data types
    new_df = new_df.astype(dtype_dict)

    # Create and fit label encoders
    for category in new_df.select_dtypes(include='object'):
        new_df[category] = label_encoders[category].transform(new_df[category])
    
    #Scale the data
    scld = scaler.transform(new_df)

    return  render_template('index.html', result=str(model.predict(scld)))

@app.route('/batch', methods=['POST'])
def batch():
    file = request.files['file']
    new_data = eval(file.read().decode('utf-8'))
    
    columns = ['restaurant type', 'rate (out of 5)', 'num of ratings', 'online_order', 'table booking', 'cuisines type','area']

    # Define data types for specific columns using a dictionary
    dtype_dict = {'rate (out of 5)': float, 'num of ratings': int}

    # Create the DataFrame with specified data types
    new_df = pd.DataFrame(new_data, columns=columns)

    # Convert specific columns to the specified data types
    new_df = new_df.astype(dtype_dict)

    # Create and fit label encoders
    for category in new_df.select_dtypes(include='object'):
        new_df[category] = label_encoders[category].transform(new_df[category])
    
    #Scale the data
    scld = scaler.transform(new_df)

    pred_df = pd.DataFrame(model.predict(scld))
    new_data = pd.DataFrame(new_data)
    new_data = pd.concat([new_data,pred_df],axis=1)
    new_data.columns = ['restaurant type','rate (out of 5)','num of ratings','online_order','table booking','cuisines type','area','avg cost']

    # Convert the DataFrame to an HTML table
    table_html = new_data.to_html(classes='table table-bordered table-striped', escape=False, index=False)

    return  render_template('index.html', table=table_html)

    # try:
    #     data = request.get_json()
    #     # Perform any required data preprocessing
    #     # Make predictions using the loaded model
    #     predictions = model.predict(data)
    #     # Return the predictions as JSON
    #     return jsonify({'predictions': predictions.tolist()})
    # except Exception as e:
    #     return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)