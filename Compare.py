#follow this guide
import time
import numpy as np
import tflite_runtime.interpreter as tflite
# Load the TensorFlow Lite model
model_path = 'mobilenet_v1_1.0_224_quant.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load the input data
input_shape = interpreter.get_input_details()[0]['shape'][1:3]
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
input_data = np.random.rand(1, *input_shape, 3).astype(np.uint8)
input_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_index, input_data)

# Run inference without the Coral Accelerator
num_runs = 1000
start_time = time.monotonic()
for i in range(num_runs):
    interpreter.invoke()
end_time = time.monotonic()
inference_time_without_accelerator = (end_time - start_time) / num_runs
print(f"Inference time without Coral Accelerator: {inference_time_without_accelerator:.2f} seconds per run")

# Load the Coral Accelerator
coral_delegate = tflite.load_delegate('edgetpu.dll') # for windows 
#coral_delegate = tflite.load_delegate('libedgetpu.so.1.0') # for Linux
model_path = "mobilenet_v1_1.0_224_quant_edgetpu.tflite" 

# Run inference with the Coral Accelerator
interpreter = tflite.Interpreter(model_path=model_path,
                                 experimental_delegates=[coral_delegate])
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_index, input_data)
start_time = time.monotonic()
for i in range(num_runs):
    interpreter.invoke()
end_time = time.monotonic()
inference_time_with_accelerator = (end_time - start_time) / num_runs
print(f"Inference time with Coral Accelerator: {inference_time_with_accelerator:.2f} seconds per run")

# Compare the inference times
performance_improvement = 100 * (inference_time_without_accelerator - inference_time_with_accelerator) / inference_time_without_accelerator
print(f"Performance improvement with Coral Accelerator: {performance_improvement:.2f}%")
