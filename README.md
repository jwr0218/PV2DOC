# PV2DOC - PresentationVideo2Document
![VL2D framework](https://github.com/jwr0218/VL2D/assets/54136688/d1fb3fa4-97ea-43f7-b3cb-3bd0cdcd4501)


### Environment ( Docker image ) 
```md
docker pull tensorflow/tensorflow:2.7.0-gpu
```

### Environment
```md
apt-get install list : 
ffmpeg=7:4.2.7-0ubuntu0.1
libgl1-mesa-glx=21.2.6-0ubuntu0.1~20.04.2
libglib2.0-0=2.64.6-1~ubuntu20.04.6
lmodern=2.004.5-6
pandoc=2.5-3build2
python-dev
tesseract-ocr=4.1.1-2build2
texlive-xetex=2019.20200218-1
```


### Dependency
```md 
pip install -r requirements.txt
```

### Download Mask-Rcnn Model (Figure & Formular Detection)

Already Uploaded Mask-RCNN model to Git. 
If you have problem to download model, you can download with below code 
```md 
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/file/d/1PTzFMJp-pF2Tt-EwPyibfj2w0KMfm9Mi/view?usp=sharing' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PTzFMJp-pF2Tt-EwPyibfj2w0KMfm9Mi" -O capstone_200_ppt.h5 && rm -rf ~/cookies.txt
```

### Activate Our Solution 

```md 
python main.py [file_name]
```


## Mask-RCNN Custom 

### code 
```md
git lfs pull
```
