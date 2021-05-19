from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import tensorflow as tf
import numpy as np
# import module1

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("hello.html")

# Emotion - 예측해서 GAN에 활용할 사진(#.jpg) 업로드 or 캡처 
@app.route("/multi_upload_emotion", methods = ['POST'])
def multi_upload_emotion():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("static/Emotion/{}.jpg".format(IDX))
        IDX += 1
    return redirect(url_for("hello"))

# GAN - 복원시킬 사진(#.jpg) 업로드
@app.route("/multi_upload_gan", methods = ["POST"])
def multi_upload_gan():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("static/GAN/{}.jpg".format(IDX))
        IDX += 1
    return redirect(url_for("hello"))

@app.route("/emotion_prediction", methods = ['POST'])
def emotion_prediction():
    # 1. 이미지 업로드
    # 2. resize
    # 3. 모델 불러와서 업로드한 이미지 학습
    # 4. 예측된 표정에 맞는 곳에 저장(ex)"static/Emotion/angry/{}.jpg")
    uploaded_files = request.files.getlist("file[]")
    index_happy = 0
    index_angry = 0
    index_neutral = 0
    for file in uploaded_files:
        filestr = file.read() # byte 단위이기 때문에 바로 file.save로 저장해서 .jpg로 보이지 않는다.
        
        detection_model_path = 'models/haarcascade_frontalface_default.xml'
        face_detection = cv2.CascadeClassifier(detection_model_path)

        #convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        frame = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE) # imread와 달리 byte 읽기
        saveFile = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # print(type(frame)) # numpy.ndarray
        face = face_detection.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        
        # print(face)
        if len(face) == 0: # 얼굴 인식 안된 사진은 Train Set에 저장되지 않도록
            continue
        face = sorted(face, reverse = True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face 
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)
        # print(roi)

        prediction = model.predict(roi) # ex) [happy, angry, neutral] -> [0.33, 0.1, 0.57]
        print(prediction)

        # {'angry': 0, 'happy': 1, 'neutral': 2}
        # Threshold를 0.5로 설정. 확률 0.5가 넘는 표정이 있으면 해당 폴더에 저장.
        if prediction[0][0] >= 0.5:
            cv2.imwrite("static/Emotion/angry/{}.jpg".format(index_angry), saveFile) # numpy.ndarray
            index_angry += 1
        elif prediction[0][1] >= 0.5:
            cv2.imwrite("static/Emotion/happy/{}.jpg".format(index_happy), saveFile)
            index_happy += 1
        elif prediction[0][2] >= 0.5:
            cv2.imwrite("static/Emotion/neutral/{}.jpg".format(index_neutral), saveFile)
            index_neutral += 1
        
    return redirect(url_for("hello"))
'''
@app.route("/happy")
def happy():
    return render_template("happy.html")

@app.route("/angry")
def angry():
    return render_template("angry.html")

@app.route("/neutral")
def neutral():
    return render_template("neutral.html")
'''
@app.route('/index')
def emotion():
    return render_template('index.html')

@app.route("/restore")
def restore():
    return render_template("restore.html")

if __name__ == "__main__":
    # face_detection = load_detection_model('models/haarcascade_frontalface_default.xml')
    model = load_model('models/model_best_0_2.h5') # model load

    # app.run(host='0.0.0.0') # 외부에서 접근가능한 서버로 만들어준다, 외부에서 접근가능하도록 하는 U
    # app.run(debug = True)
    # app.run(host='0.0.0.0') # defalut port = 5000
    