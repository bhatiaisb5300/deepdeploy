from django.shortcuts import render
# from django.conf import settings
from deepdeploy.settings import model
from base64 import b64encode
import cv2
import numpy as np
# Create yor views here.
def index(request):
    if request.method=='POST':
        if request.FILES:
            inImg = request.FILES["file"]
            response = inImg.read()
            img = cv2.imdecode(np.fromstring(response, np.uint8), cv2.IMREAD_COLOR)
            img = (cv2.resize(img, (150,150))/255).reshape(1,150,150,3)
            print(type(img))
            prediction = model.predict(img)

            if prediction < 0.5:
                var = 'cat'
                print(prediction)
            else:
                print(prediction)
                var = 'dog'
            return render(request, "result.html",{'prediction':var})
    return render(request,'index.html')


def result(request):
    return render(request,'result.html')
