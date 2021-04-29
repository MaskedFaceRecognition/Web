from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

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
        filestr = file.read()
        #convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, dsize = (48, 48))
        img = img[..., np.newaxis]
        img = img[..., None]
        img = img.reshape(-1, 48, 48, 1)
        img = tf.reshape(img, (-1, 48, 48, 1))

        prediction = model.predict(img) # [happy, angry, neutral]가 [0.33, 0.1, 0.57] 이런 식으로 나옴.

        # {'angry': 0, 'happy': 1, 'neutral': 2}
        # Threshold를 0.5로 설정하였다. 3개 중 확률이 0.5가 넘는 해당 값이면 각각 폴더에 저장되는 걸로.
        if prediction[0][0] >= 0.5:
            file.save("static/Emotion/angry/{}.jpg".format(index_angry))
            index_angry += 1
        elif prediction[0][1] >= 0.5:
            file.save("static/Emotion/happy/{}.jpg".format(index_happy))
            index_happy += 1
        elif prediction[0][2] >= 0.5:
            file.save("static/Emotion/neutral/{}.jpg".format(index_neutral))
            index_neutral += 1
        
        

    return redirect(url_for("hello"))

@app.route("/happy")
def happy():
    return render_template("happy.html")

@app.route("/angry")
def angry():
    return render_template("angry.html")

@app.route("/neutral")
def neutral():
    return render_template("neutral.html")

@app.route('/index')
def emotion():
    return render_template('index.html')

@app.route("/restore")
def restore():
    return render_template("restore.html")

if __name__ == "__main__":
    # face_detection = load_detection_model('models/haarcascade_frontalface_default.xml')
    model = load_model('models/model_best_0_2.h5') # model load
    # app.run(host='0.0.0.0') # 외부에서 접근가능한 서버로 만들어준다, 외부에서 접근가능하도록 하는 URL은?
    app.run()