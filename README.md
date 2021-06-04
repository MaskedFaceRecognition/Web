# 얼굴 인식 및 복원 Web

> AWS EC2 이용 ubuntu 20.04 OS
> 
> mod_wsgi 이용 Apache2와 Flask 연동


- 웹 디렉토리 내에는 apache2와 flask 연동 작업을 위한 web.wsgi 파일과 웹 서비스의 세부 구현 동작을 수행하는 flask의 app.py로 구성되어 있다. 
- 세부 디렉토리에는 logs, src_upgraded, static, templates가 있다.
- logs 디렉토리 내의 .log파일로 서버 실행 중 에러 및 접속에 대한 로그를 확인할 수 있다.
- models 디렉토리에는 얼굴을 확인하고 감정 분류를 수행하기 위해 앞서 구성되어 있는 face
- detection model과 감정 분류 모델이 저장되어 있다. src_upgraded 디렉토리는 glcic layer가 정의되어 있는 layer.py, test, train 파일을 불러오는 load.py, mask detector의 모델인 mask_detector.model, check point를 저장하는 backup 디렉토리, gan 모델에서 얼굴 인식을 위한 모델이 저장되어 있는 face_detector 디렉토리, layer.py에서 정의한 layer들로 network를 구성한 network.py가 있다. 
- static 디렉토리는 감정 분류 후 저장되는 Emotion 디렉토리와, GAN 학습을 위해 batch 파일로 만들어주기 위한 npy 디렉토리, GAN 학습 후 결과물이 저장되는 output 디렉토리로 구성되어 있다. 
- templates에는 home 화면인 hello2.html, 마스크를 쓰지 않은 사진 업로드 화면인 upload_nonmasked.html, 마스크를 쓴 사진 업로드 화면인 upload_masked.html과 css, js파일로 구성되어 있다.
![image](https://user-images.githubusercontent.com/43158502/120748677-25ef0e00-c53e-11eb-9f7f-397dadde019a.png)

## 진행 내용 정리
- 일자 별 진행내용 정리


