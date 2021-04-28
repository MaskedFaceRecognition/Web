from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import cv2
from scipy import misc


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


@app.route("/emotion_classification", methods = ['POST'])
def emotion_prediction():
    # 1. 업로드 이미지 reshape
    # 2. 모델 불러와서 업로드 이미지 학습
    # 3. 예측된 표정에 맞는 곳에 저장(GAN/happy, GAN/sad ..etc)
    if request.method == 'POST':
        file = request.files['static/Emotion/image1.jpg'] # 일단 파일 1개
        if not file:
            return render_template('index_html', label = 'No Files')
    
        img = misc.imread(file)
        img = img[:, :, :3]
        img = img.reshape(-1, 1)

        prediction = model.predict(img)

        label = str(np.squeeze(prediction))

        if label == '10': label = '0'

        return render_template('index_html', label = label)


@app.route("/happy")
def happy():
    return render_template("happy.html")

@app.route("/angry")
def angry():
    return render_template("angry.html")

@app.route("/neutral")
def neutral():
    return render_template("neutral.html")

@app.route("/restore")
def restore():
    return render_template("restore.html")


if __name__ == "__main__":
    model = load_model('models/model_best_0_2.h5') # model load
    app.run()