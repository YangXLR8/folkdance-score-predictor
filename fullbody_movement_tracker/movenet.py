import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import wget
import os
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

import fullbody_movement_tracker.helperVis as hv

class Detector:
    def __init__(this, isTflite: bool):
        this.name = type(this).__name__
        this.models_dir: str = os.path.join(os.path.dirname(__file__), '..', 'model')
        this.output_dir: str = os.path.join(os.path.dirname(__file__), '..', 'output')

        this.model_file: str = ''
        this.isTflite: bool = isTflite
        this.inferenceMethod: Callable = None
        this.input_size: int = None
     

    def doImageInference(this, image_filename: str):
        if this.inferenceMethod is None:
            raise Exception(f"model not loaded. do {this.name}.loadModel(model_name) to load model")
        # Load the input image.
        image = tf.io.read_file(image_filename)
        image = tf.image.decode_image(image)

        inference = this.inferenceMethod(image)

        # save visualization
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(
            display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = hv.draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), inference)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(output_overlay)
        plt.savefig(os.path.join(this.output_dir, 'output'))

        # Return the model inference.
        return inference

    def loadModel(this, model_name: str):
        """
        This function loads the specified Movenet model based on the given model name and returns a function that can be used to perform inference on images.

        Parameters:
        this (MovenetGetter): The instance of the MovenetGetter class.
        model_name (str): The name of the Movenet model to load. It should be one of the following:
            - movenet_lightning_f16
            - movenet_thunder_f16
            - movenet_lightning_int8
            - movenet_thunder_int8
            - movenet_lightning
            - movenet_thunder

        Returns:
        function: A function that takes an input image as a numpy array and returns the predicted keypoint coordinates and scores.
        """
        if this.isTflite:    
            if "movenet_lightning_f16" in model_name:
                this.model_file = os.path.join(this.models_dir, "movenet_lightning_f16.tflite")
                if (not os.path.exists(this.model_file)):
                    print("Downloading model " + model_name)
                    wget.download("https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite", this.model_file)
                this.input_size = 192
            elif "movenet_thunder_f16" in model_name:
                this.model_file = os.path.join(models_dir, "movenet_thunder_f16.tflite")
                if (not this.path.exist(this.model_file)):
                    print("Downloading model " + model_name)
                    wget.download("https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite", this.model_file)
                this.input_size = 256
            elif "movenet_lightning_int8" in model_name:
                this.model_file = os.path.join(models_dir, "movenet_lightning_int8.tflite")
                if (not this.path.exist(this.model_file)):
                    print("Downloading model " + model_name)
                    wget.download("https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite", this.model_file)
                this.input_size = 192
            elif "movenet_thunder_int8" in model_name:
                this.model_file = os.path.join(models_dir, "movenet_thunder_int8.tflite")
                if (not this.path.exist(this.model_file)):
                    print("Downloading model " + model_name)
                    wget.download("https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite", this.model_file)
                this.input_size = 256
            else:
                raise ValueError("Unsupported model name: %s" % model_name)


            def movenet(input_image):
                """Runs detection on an input image.

                Args:
                input_image: A [1, height, width, 3] tensor represents the input image
                    pixels. Note that the height/width should already be resized and match the
                    expected input resolution of the model before passing into this function.

                Returns:
                A [1, 1, 17, 3] float numpy array representing the predicted keypoint
                coordinates and scores.
                """
                # Resize and pad the image to keep the aspect ratio and fit the expected size.
                input_image = np.expand_dims(input_image, axis=0)
                input_image = tf.image.resize_with_pad(input_image, this.input_size, this.input_size)
                # input_image = np.resize(input_image, [1, this.input_size, this.input_size, 3])

                # Initialize the TFLite interpreter
                interpreter = tf.lite.Interpreter(model_path=this.model_file)
                interpreter.allocate_tensors()

                # TF Lite format expects tensor type of uint8.
                input_image = tf.cast(input_image, dtype=tf.uint8)
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

                # Invoke inference.
                interpreter.invoke()

                # Get the model prediction.
                keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
                return keypoints_with_scores

        else:
            if "movenet_lightning" in model_name:
                module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
                this.input_size = 192
            elif "movenet_thunder" in model_name:
                module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
                this.input_size = 256
            else:
                raise ValueError("Unsupported model name: %s" % model_name)

            def movenet(input_image):
                """Runs detection on an input image.

                Args:
                input_image: A [1, height, width, 3] tensor represents the input image
                    pixels. Note that the height/width should already be resized and match the
                    expected input resolution of the model before passing into this function.

                Returns:
                A [1, 1, 17, 3] float numpy array representing the predicted keypoint
                coordinates and scores.
                """
                model = module.signatures['serving_default']

                # SavedModel format expects tensor type of int32.
                input_image = tf.cast(input_image, dtype=tf.int32)

                # Run model inference.
                outputs = model(input_image)

                # Output is a [1, 1, 17, 3] tensor.
                keypoints_with_scores = outputs['output_0'].numpy()
                return keypoints_with_scores

        this.inferenceMethod = movenet

        

        