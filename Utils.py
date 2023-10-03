# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:07:24 2021

@author: Seung kyu Hong
"""
import whisper

from mdutils.mdutils import MdUtils
import pypandoc
from mdutils import Html
import boto3
import time
import urllib,json,requests
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract as pt
from pytesseract import Output
import cv2,re
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
#from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
#import gensim
import mrcnn
import mrcnn.config
#import mrcnn.model as MD
import mrcnn.visualize
import cv2
import os
import numpy
import numpy as np 
from PIL import Image, ImageDraw
from mrcnn import model
from moviepy.editor import VideoFileClip
from IPython.display import Audio
import gensim

model_whisper = whisper.load_model("base")

CLASS_NAMES = ['BG', 'figure','formula']



class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)


def load_model():
    model1 = model.MaskRCNN(mode="inference",
                                    config=SimpleConfig(),
                                    model_dir=os.getcwd())
    model1.load_weights(filepath="capstone_200_ppt.h5",
                       by_name=True)
    return model1


def merge_boxes(results_rois,results_masks):
    #line = len(results_rois)
    boxes = list()
    for box in results_rois:
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]

        coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]

        boxes.append(coors)

    size = list(results_masks.shape[:2])
    size.append(3)

    stencil1 = numpy.zeros(size).astype(np.dtype("uint8"))
    stencil2= numpy.zeros(size).astype(np.dtype("uint8"))

    color = [255, 255, 255]

    for i in range(len(boxes)):
        stencil1 = numpy.zeros(size).astype(np.dtype("uint8"))

        contours = [numpy.array(boxes[i])]
        cv2.fillPoly(stencil1, contours, color)


        for j in range(i+1,len(boxes)):
            stencil2= numpy.zeros(size).astype(np.dtype("uint8"))
            contours = [numpy.array(boxes[j])]
            cv2.fillPoly(stencil2, contours, color)


            intersection = np.sum(numpy.logical_and(stencil1, stencil2))
        
            if intersection > 0:
                xmin = min(boxes[i][0][0],boxes[j][0][0])
                ymin = min(boxes[i][0][1],boxes[j][0][1])
                xmax = max(boxes[i][2][0],boxes[j][2][0])
                ymax = max(boxes[i][2][1],boxes[j][2][1])

                '''
                coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]
                '''
                #print(" {},{} INTERSECTION : {}".format(i,j,np.sum(intersection)))

                results_rois[i] = [ymin,xmin,ymax,xmax]
                arr = np.delete(results_rois,j,0)
                
                return merge_boxes(arr,results_masks)

    return results_rois


def measureing(data, y_hap):
    average_score = silhouette_score(data, y_hap)

    #print(average_score)
    #print('Silhouette Analysis Score:',average_score)
    return average_score



def cluster_hierarchy(data):
    lst = []
    for i in range(1,5):
        try:
            hc = AgglomerativeClustering(n_clusters=i , linkage='average')

            y_hc = hc.fit_predict(data)
            lst.append(measureing(data,y_hc))
        except:
            pass
    try:
        hipher_para = lst.index(max(lst))+2
        print("hipher_parameter : ",hipher_para)
    except:
        hipher_para=1

    hc = AgglomerativeClustering(n_clusters=hipher_para)

    y_hc = hc.fit_predict(data)

    a = y_hc.reshape(-1, 1)

    return a , y_hc

def mse(A, B):
    err = np.sum((A.astype("float") - B.astype("float")) ** 2)
    err /= float(A.shape[0] * A.shape[1])


def extract_Figures(model,pil_image):
    
    image=np.array(pil_image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r = model.detect([image], verbose=0)
    r = r[0]
    merged = merge_boxes(r['rois'],r['masks'])
    extract_imgs = list()
    for i in merged:

        
        #  cropped_img = img[y: y + h, x: x + w]
        cropped_img = image[i[0]:i[2], i[1]: i[3]]
        extract_imgs.append(Image.fromarray(cropped_img))
    image = Image.fromarray(image)
    for i in merged:
        #print(i)
        shape = [(i[1], i[0]), (i[3],i[2])]
        
        img1 = ImageDraw.Draw(image)
        img1.rectangle(shape, fill ="#FFFFFF")
    
    return extract_imgs , image



def extract_from_video(model,video_dir):
    cnt = 1

    cap = cv2.VideoCapture(video_dir)

    FPS = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / FPS

    second = 1
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, frame = cap.read()

    #plt.imshow(frame)
    #plt.show()
    pil_Image = Image.fromarray(frame)


    image_deleted_list = list()
    image_cropped_list = list()
    #print(cnt)
    image_cropped , image_deleted = extract_Figures(model,pil_Image)
    image_deleted_list.append(image_deleted)
    image_cropped_list.append(image_cropped)

    num = 0
    increase_width = 3

    while success and second <= duration:
        num += 1
        second += increase_width
        x1 = frame
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1500)
        success, frame = cap.read()
        x2 = frame

        x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
        try:

            x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
        except:
            continue

        diff = cv2.subtract(x1, x2)
        result = not np.any(diff)
        s = ssim(x1, x2)

        if s < 0.95:
            # 바뀐경우 flame 을 바꿔야 함
            pil_Image = Image.fromarray(frame)
            #plt.imshow(frame)
            #plt.show()
            image_cropped , image_deleted  = extract_Figures(model,pil_Image)
            image_deleted_list.append(image_deleted)
            image_cropped_list.append(image_cropped)

            cnt += 1
    
    
    columns = ['left','top','width','height','text','word_count','slid_num']
    data = pd.DataFrame(columns= columns)

    for j in range(len(image_deleted_list)):
        image = image_deleted_list[j]
        #img = cv2.imread('C:/Users/Seung kyu Hong/Desktop/temp/' + str(p) + '.png')
        #img = cv2.imread(')
        #df = pt.image_to_data(img,output_type=Output.DATAFRAME)
        df = pt.image_to_data(image,output_type=Output.DATAFRAME)
        #custom_config = r'--oem 3 --psm 6'
            
        for i in range(len(df)):
            if df['text'][i]==' ':
                df=df.drop(i,axis=0)
        df = df.reset_index()
        count = 1
        df['line'] = -1
        for i in range(len(df)):        
            if df['conf'][i]>-1:
                df['line'][i] = count
                try:
                    if df['conf'][i+1]>-1:
                        continue
                except:
                    pass
                else:
                    count += 1
            
        df = df.dropna()           
        try:

            df['text'] = df.apply(lambda x : x['text']+" " , axis = 1 )
        except ValueError:
            continue
        final = pd.DataFrame(columns = ['left','top','width','height','text','word_count','slid_num'])
        
        
        final['left'] = df.groupby(df['line'])['left'].min()
        final['top'] = df.groupby(df['line'])['top'].mean()
        final['width'] = df.groupby(df['line'])['width'].sum()
        final['height'] = df.groupby(df['line'])['height'].mean()
        final['text'] = df.groupby(df['line'])['text'].sum()
        
        final['word_count'] = final.apply(lambda x: len(x['text'].replace(' ','')), axis = 1)
        final['slid_num'] = j+1
        #print(final)
        if final.size>1:
            data = pd.concat([data,final],axis=0)


    data = data.reset_index(drop=True)
        
    
    for i in range(len(data)):
        data['text'][i] = data['text'][i].strip()
        
    data = data.where(data['text']!="")
    data = data.dropna()
    data['word_size'] = data['width']/data['word_count']
    slid_lst = data['slid_num'].unique()
    lst_result = np.ndarray([])
    
    for i in slid_lst:
        try:
            data_for_test = data[data['slid_num'] == i]
        except KeyError:
            #print("end =============")
            break
        
        if len(data_for_test) ==1:
            a = [[0]]
            y_hap = [0]
            lst_result = np.append(lst_result, a)
            continue
        data_for_train = data_for_test[['left','word_size','height']]
        
        linked = linkage(data_for_train)
        a, y_hap = cluster_hierarchy(data_for_train)
        
        lst_result = np.append(lst_result, a)
        
    data['cluster'] = lst_result[1:].reshape(-1,1)
        
    data['cluster'] = data['cluster'].apply(lambda x: int(x))
    return image_deleted_list , image_cropped_list,data


def make_Audio(pre_dir,video_name):

        # 비디오 파일 경로
    #video_path = 'your_video_file.mp4'

    # 비디오 클립 생성
    video_path = pre_dir + video_name
    video_clip = VideoFileClip(video_path)

    # 오디오 추출
    audio_clip = video_clip.audio

    # 오디오 저장
    output_folder = 'tmp_audio/'+video_name.split('.')[0]+'/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 폴더가 없으면 생성

    audio_output_path = output_folder + video_name.split('.')[0] + '.wav'
    audio_clip.write_audiofile(audio_output_path)

    # 클립들 해제
    video_clip.close()
    audio_clip.close()
    return audio_output_path
def transcribe(audio):
    # load audio and pad/trim it to fit 30 seconds
    # audio = whisper.load_audio(audio)
    # audio = whisper.pad_or_trim(audio)

    # # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio).to(model_whisper.device)

    # # detect the spoken language
    # _, probs = model_whisper.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # # decode the audio
    # options = whisper.DecodingOptions()
    # result = whisper.decode(model_whisper, mel, options)
    # print(result)
    model = whisper.load_model('base')
    result = model.transcribe(audio)

    # 영문으로 자동 변환을 하려면 task='translate' 추가
    # model.transcribe('audio.mp3', task='translate')

    
    #return result.text
    return result['text']

def STT_summar(pre_dir , video_name):
    audio_output_path = make_Audio(pre_dir,video_name)
    #audio_dir = 'tmp_audio/'+video_name.split('.')[0]+'/' + video_name.split('.')[0]+'.wav'
    text = transcribe(audio_output_path)
    
    stt_text = gensim.summarization.summarize(text,ratio=0.3)
    return stt_text

def markdown(img_name,data,stt,imgCropped):
    #s3 = boto3.resource('s3')
    #bucket = s3.Bucket('capstone2021itm')
    image_dir = 'image/'+img_name+'/images'
    os.makedirs(image_dir, exist_ok=True)
    image_dir +='/'
    CAPITAL = 'QWERTYUIOPASDFGHJKLZXCVBNM0123456789'
    CHAR = '!@#$%^&*()-+=_?/><.,`~[]{}*'
    i_name = img_name.split('.')
    file_name = i_name[0]+'_documents'
    mdFile = MdUtils(file_name = file_name,title="Markdown")
    mdFile.new_header(level=1,title=img_name+' OCR')
    numCluster = max(data['cluster'])
    count = 0
    check_num = 0 
    for j in range(1,len(data['slid_num'].unique())+1):
        
        if data[data['slid_num']==j].size < 1:
            continue
        
        #mdFile.new_header(level=2,title='page '+str(j))
        #print(data[data['slid_num']==j]['text'])
        #print('\n =================MARKDOWON================= \n ')
        #print(data[data['slid_num']==j])
        #print("COUNT  : ",count)

        title = data[data['slid_num']==j]['text'][count]
        #print(title)

        #mdFile.new_header(level=3,title=title)
        text = ''
        lst = []
        cnt = 1
        #print(title)
        for i,t in enumerate(zip(data[data['slid_num']==j]['text'],data[data['slid_num']==j]['cluster'])):
            print(t)
            
            if t[0]==title:
                
                continue
            '''
            if t[0][0] in CHAR:    
                continue
            '''
            if t[1]==numCluster-1:
                text+= t[0]+' '
                if t[0][0] not in CAPITAL:
                    lst.append(text+'\n')
                    text= ''
                    
                    continue
                else:
                    if i+1<len(data[data['slid_num']==j]['text']) and data[data['slid_num']==j]['cluster'][count+1]!=t[1]:
                        lst.append(t+'\n')
            if t[1]==numCluster-2:
                lst.append([t[0]]+'\n')
        #print(lst)
        count += len(data[data['slid_num']==j]['text'])    
        
        if len(lst)<1:
            check_num+=1
            continue            
        
        mdFile.new_header(level=2,title='page-'+str(j-check_num)+'\n')
        
        mdFile.new_header(level=3,title=title+'\n')
        
        mdFile.new_list(lst)
        
        
        for image in imgCropped[j-1]:
            
            if image:
                image.save('./image/'+img_name+'/images/'+str(j)+'-'+str(cnt)+'.png',format='PNG')
                #bucket.upload_file('image/'+img_name+'images/'+str(j)+'-'+str(cnt)+'.png',img_name+'/'+'page'+str(j)+'-'+str(cnt)+'.png')
                
                #cv2.imwrite(str(image)+'.png',image)
                
                mdFile.new_line(mdFile.new_inline_image(text='page '+str(j)+'-'+str(cnt), path='./image/'+img_name+'/images/'+str(j)+'-'+str(cnt)+'.png'))
                cnt += 1
        mdFile.new_line("\pagebreak")
        mdFile.new_line('\n') 

    mdFile.new_line("\pagebreak") 
    mdFile.new_header(level=1,title=img_name+' STT')
    mdFile.new_paragraph(text=stt)
    mdFile.create_md_file()
    
    output = pypandoc.convert_file('./'+file_name+'.md', 'pdf', outputfile='./'+file_name+".pdf")


def main_solution(model,pre_dir , file_name):


    imgDeleted,imgCropped,data = extract_from_video(model,pre_dir+file_name)
    

    stt = STT_summar(pre_dir,file_name)
    print('STT : ',stt)
    print(type(stt))
    #stt = "testing / we are testing this system. Is this work?"
    markdown(file_name,data,stt,imgCropped)