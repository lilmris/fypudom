from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf  # Ensure TensorFlow is installed
from tensorflow import keras
from tensorflow import load_model


# Load your trained TensorFlow model (replace with your actual loading logic)
model = keras.models.load_model('model.h5')

def predict_virus_class(sequence_string):
   # Generate an array of random integers between 1 and 3055 (inclusive)
   sequence_string = np.random.randint(1, 300, size=16383)

   print(sequence_string)

   # Preprocess the sequence string
   max_length = 16383
   # Use 'constant_values' to specify padding value for 'constant' mode
   sequence_padded = np.pad(sequence_string, (0, max_length - len(sequence_string)), mode='constant', constant_values=0) 
   sequence_padded = sequence_padded.reshape(1, -1, 1) # Reshape to add batch and feature dimension

   # Make prediction
   predicted_probabilities = model.predict(sequence_padded)
   predicted_class = np.argmax(predicted_probabilities, axis=1)[0]
   print("Predicted Class:", predicted_class)
   return predicted_class

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  # Get the sequence data from the request
  data = request.get_json()
  if not data or 'sequence' not in data:
    return jsonify({'error': 'Missing sequence data'}), 400

  sequence_string = data['sequence']

  # Make prediction and return the result
  predicted_class = predict_virus_class(sequence_string)
  return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
  # Run the Flask app (use port appropriate for Heroku deployment)
  app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

