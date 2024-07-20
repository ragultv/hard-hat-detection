from ultralytics import YOLO
import gradio as gr
# Create a new YOLO model from scratch
model = YOLO("best.pt")
#define function
def detection(image):
    results = model(image)
    return results[0].plot()
iface = gr.Interface(fn=detection, inputs="image", outputs="image")
#launch the interface
iface.launch()
