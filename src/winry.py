import pylsl
import numpy as np
import pywt
import joblib
<<<<<<< HEAD
from tensorflow.keras.models import load_model
=======
import pandas as pd
>>>>>>> fdbb1b4becdbba4eeea8bcbbd6ae525e5330d099

# Load the trained machine learning model
model_file_path = 'trained_model.h5'
model = load_model(model_file_path)

# Load the fitted scaler
scaler_file_path = 'scaler.joblib'
scaler = joblib.load(scaler_file_path)

# Define the best wavelet transform parameters (as determined during training)
best_wavelet = 'db4'  # The wavelet used during training
best_level = 5  # The level used during training
expected_input_shape = model.input_shape[1]  # The expected input shape by the model

# Define a function to perform wavelet transformation
def wavelet_transform(data, wavelet=best_wavelet, level=best_level):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = np.concatenate(coeffs, axis=-1)
    return coeffs

# Function to pad or truncate data to the expected input size
def pad_or_truncate(data, size):
    if len(data) > size:
        return data[:size]
    elif len(data) < size:
        return np.pad(data, (0, size - len(data)), 'constant')
    else:
        return data

# Function to acquire data from the LSL stream and classify it
def classify_eeg_data():
    print("Looking for an EEG stream...")
    brain_stream = pylsl.resolve_stream("name", "AURA_Power")
    
    if not brain_stream:
        print("No EEG stream found.")
        return
    
    brain_inlet = pylsl.StreamInlet(brain_stream[0])
    info = brain_inlet.info()
    sample_rate = info.nominal_srate() 
    print(f"LSL stream found with sample rate: {sample_rate} Hz")
    brain_inlet.open_stream()

    # Accumulate data for a given period (e.g., 5 seconds)
    accumulation_period = 5.0  # seconds
    num_samples_to_accumulate = int(sample_rate * accumulation_period) 
    accumulated_samples = []

    # Create the LSL outlet for classification labels
    label_info = pylsl.StreamInfo('Riza_Hawkeye', 'Markers', 1, 0, 'string', 'myuniquelabelid')
    label_outlet = pylsl.StreamOutlet(label_info)

    while True:
        # Read a sample from the inlet
        sample, timestamp = brain_inlet.pull_sample()

        # Accumulate the sample
        accumulated_samples.append(sample)
        
        if len(accumulated_samples) >= num_samples_to_accumulate:
            # Convert accumulated samples to a NumPy array
            accumulated_samples_np = np.array(accumulated_samples)
            
            # Print the features before scaling
            print(f"Features before scaling:\n{accumulated_samples_np.shape}")

            # Standardize the accumulated samples
            accumulated_samples_scaled = scaler.transform(accumulated_samples_np)

            # Apply wavelet transform to the accumulated samples
            samples_wavelet = np.array([wavelet_transform(sample) for sample in accumulated_samples_scaled])
            
            # Reshape the samples to match the input shape expected by the model
            samples_wavelet = samples_wavelet.reshape(1, -1)
            
            # Pad or truncate the data to match the expected input shape
            samples_wavelet = pad_or_truncate(samples_wavelet[0], expected_input_shape)
            
            # Reshape again to ensure it matches the input shape
            samples_wavelet = samples_wavelet.reshape(1, -1)
            
            # Print the shape after wavelet transform and reshaping
            print(f"Shape after wavelet transform and reshaping: {samples_wavelet.shape}")

            # Classify the sample using the loaded model
            probabilities = model.predict(samples_wavelet)
            
            # Convert probabilities to class labels
            label = np.argmax(probabilities, axis=1)
            
            # Print the classification label
            print(f"Classified label: {label[0]}")

            # Send the classification label via LSL
            label_outlet.push_sample([str(label[0])])
            print(f"Sent label via LSL: {label[0]}")

            # Clear the accumulated samples
            accumulated_samples = []

# Run the classification function
if __name__ == "__main__":
    classify_eeg_data()
