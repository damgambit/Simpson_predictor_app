# Simpson_predictor_app
A simple app that predict which Simpson character you make it see!
Here is an example of it in action:
<script src="http://vjs.zencdn.net/4.0/video.js"></script>

<video id="pelican-installation" class="video-js vjs-default-skin" controls
preload="auto" width="683" height="384" data-setup="{}">
<source src="/static/screencasts/pelican-installation.mp4" type='video/mp4'>
</video>

### Insides on the deep learning model I used to train my image classifier
Deep Learning model repository: https://github.com/damgambit/simpsons_characters-classification

# App 

### Pre-requisites
- npm
- react-native

### Setup
```
cd simpsons_app/
npm install 
react-native link
react-native run-ios
```


# Predictions API

### Pre-requisites
- Python==2.7

### Setup
```
cd simpson-predictions-api
pip install -r requirements.txt
python app.py
```




