from ultralytics import YOLO
import cv2
from flask import Flask,request,send_file
import io
from PIL import Image
import numpy as np
# Create a new YOLO model from scratch
model = YOLO("best.pt")

app=Flask('__name__')

def detection(image):
    results=model(image)
    
@app.route('/detect',methods=['POST'])
def detect():
    if request.method=='POST':
        image=request.files.get['file']
        converted=Image.open(image)
        imgarr=np.array(converted)
        results=detection(imgarr)

        result_img = results[0].plot()
        result_img_pil = Image.fromarray(result_img)
        img_io = io.BytesIO()
        result_img_pil.save(img_io, 'JPEG')
        img_io.seek(0)
        # Return the image with predictions
        return send_file(img_io, mimetype='image/jpeg')

if __name__=='__main__':
    app.run(debug=True)