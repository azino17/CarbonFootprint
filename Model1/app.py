from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# Load the model and LabelEncoder
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    emissions = data.get('emissions')
    cost = data.get('cost')
    
    if emissions is None or cost is None:
        return jsonify({'error': 'Missing input data'}), 400
    
    # Ensure that strategies are properly encoded
    strategies = le.classes_
    
    # Prepare input data for all strategies
    input_data = pd.DataFrame({
        'Emissions (tonnes)': [emissions] * len(strategies),
        'Cost (USD)': [cost] * len(strategies),
        'Strategy': le.transform(strategies)
    })
    
    # Predict effectiveness
    predicted_effectiveness = model.predict(input_data)
    
    # Find the best strategy
    best_index = np.argmax(predicted_effectiveness)
    best_strategy = strategies[best_index]
    best_effectiveness = predicted_effectiveness[best_index]
    
    return jsonify({
        'best_strategy': best_strategy,
        'best_effectiveness': best_effectiveness
    })

if __name__ == '__main__':
    app.run(port=5000,debug=True)
