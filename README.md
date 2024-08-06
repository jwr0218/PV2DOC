# PV2DOC - PresentationVideo2Document
![VL2D framework](https://github.com/jwr0218/VL2D/assets/54136688/d1fb3fa4-97ea-43f7-b3cb-3bd0cdcd4501)


### Environment ( Docker image ) 
```md
docker pull pytorch:pytorch
```

### Environment
```md
apt-get install -y --no-install-recommends \
        ffmpeg=7:4.2.7-0ubuntu0.1 \
        libgl1-mesa-glx=21.2.6-0ubuntu0.1~20.04.2 \
        libglib2.0-0=2.64.6-1~ubuntu20.04.7 \
        lmodern=2.004.5-6 \
        python-dev \
        tesseract-ocr=4.1.1-2build2 \
        texlive-xetex=2019.20200218-1 \
```


### Dependency
```md 
pip install -r requirements.txt
```

### Download Mask-Rcnn Model (Figure & Formular Detection)

Already Uploaded Mask-RCNN model to Git. 
If you have problem to download model, you can download with below address
```md
MRCNN model : 
https://drive.google.com/file/d/1PTzFMJp-pF2Tt-EwPyibfj2w0KMfm9Mi/view?usp=sharing
YOLO Model : 
https://drive.google.com/file/d/1xwnx3B290BWID0JfhJU87ya82yC0mnUT/view?usp=sharing
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
