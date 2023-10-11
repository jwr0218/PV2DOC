# PV2DOC - PresentationVideo2Document
![VL2D framework](https://github.com/jwr0218/VL2D/assets/54136688/d1fb3fa4-97ea-43f7-b3cb-3bd0cdcd4501)


### Environment ( Docker image ) 
```md
docker pull tensorflow/tensorflow:2.7.0-gpu
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
