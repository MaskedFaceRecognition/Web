from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import sys
import cv2
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("hello.html")

# Emotion - 예측해서 GAN에 활용할 사진 업로드 or 캡처
@app.route("/multi_upload_emotion", methods = ["POST"])
def multi_upload_emotion():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("static/Emotion/{}.jpeg".format(IDX))
        IDX += 1
    return redirect(url_for("hello"))

# GAN - 복원시킬 사진 업로드
@app.route("/multi_upload_gan", methods = ["POST"])
def multi_upload_gan():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("static/gan/{}.jpeg".format(IDX))
        IDX += 1
    return redirect(url_for("hello"))

@app.route("/happy")
def happy():
    return render_template("happy.html")

@app.route("/sad")
def sad():
    return render_template("sad.html")

@app.route("/angry")
def angry():
    return render_template("angry.html")

@app.route("/neutral")
def neutral():
    return render_template("neutral.html")



@app.route("/restore")
def restore():
    return render_template("restore.html")


@app.route("/emotion_classification")
def emotion_prediction():
    # 1. 업로드 이미지 reshape
    # 2. 모델 불러와서 업로드 이미지 학습
    # 3. 예측된 표정에 맞는 곳에 저장(GAN/happy, GAN/sad ..etc)
    if request.method == 'POST':
        file = request.files['static/Emotion']
    '''
         따로 처리 필요
        # if not file:
            # return render_template('index_html', label = 'No Files')
    '''
    src = cv2.imread(file)
    img = cv2.resize(src, dsize = (48, 48))
    
    prediction = model.predict(img)
    


if __name__ == "__main__":
    # 모델 로드
    model = load_model('model_bestweight_0_2.h5')
    app.run(host='0.0.0.0')
