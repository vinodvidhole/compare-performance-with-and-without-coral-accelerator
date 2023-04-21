import time
import numpy as np
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Load the TensorFlow Lite model
model_path = "mobilenet_v1_1.0_224_quant.tflite"
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load an image and preprocess the input data
img = tf.keras.preprocessing.image.load_img('test.jpg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
input_data = np.array(img_array, dtype=np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference without the Coral accelerator
start_time = time.time()
interpreter.invoke()
end_time = time.time()
print("Inference time without Coral accelerator:", end_time - start_time)

# Load the Edge TPU delegate
coral_delegate = load_delegate('libedgetpu.so.1.0')
#model_path = "mobilenet_v1_1.0_224_quant_edgetpu.tflite" 
# Reload the TensorFlow Lite model with the Edge TPU delegate
interpreter = Interpreter(model_path=model_path, experimental_delegates=[coral_delegate])
interpreter.allocate_tensors()

# Set the input tensor again
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference with the Coral accelerator
start_time = time.time()
interpreter.invoke()
end_time = time.time()
print("Inference time with Coral accelerator:", end_time - start_time)
