import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import plotly.express as px
from drought_det_train import transform_user_img, make_fig, get_dataframe_data 
import transform_user_img, make_fig, get_dataframe_data

MODEL_PATH = "path/to/model"
@gradio.utils.cache
def load_model(model_path):
return tf.saved_model.load(model_path)

model = load_model(MODEL_PATH)
def predict(image):
# Transform user image
image = transform_user_img(image)
# Predict from the model
prediction = model(image)

return prediction
labels = ["Drought", "Drought Risk", "Possible Drought Risk", "No Drought"]
explanations = [
"Looks like your region is likely suffering from drought ",
"Looks like your region is at risk of drought. ",
"Looks like your region is not suffering from a drought.",
"Looks like your region is healthy"
]

image_input = gr.inputs.Image(label="Choose a file")
output_text = gr.outputs.Textbox(label="Drought Prediction")
output_plot = gr.outputs.Plot(label="Drought Likelihood")

def inference(image):
# Perform prediction
prediction = predict(image)
result = np.argmax(prediction)
# Prepare output
prediction_label = labels[result]
explanation = explanations[result]

# Prepare dataframe for plotting
df = get_dataframe_data(prediction)
x = df.iloc[:, 2]
y = df.iloc[:, 1]
fig = make_fig(df, x, y)

return prediction_label, explanation, fig
gr.Interface(inference, inputs=image_input, outputs=[output_text, output_text], title="Drought conditions in Ethiopia",
description="Applying deep learning and computer vision for drought resilience, using satellite images and human expert labels to detect drought conditions in Ethiopia").launch()
