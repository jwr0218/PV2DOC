# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:07:24 2021

@author: Seung kyu Hong
"""
import whisper

from mdutils.mdutils import MdUtils
# import pypandoc
import aspose.words as aw
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
from moviepy.editor import VideoFileClip
from IPython.display import Audio
import nltk
nltk.download('punkt')
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer




CLASS_NAMES = ['BG', 'figure','formula']


import torch 




class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)


def load_model():
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s_best.pt', force_reload=True)
    # Set the model to evaluation mode
    model.eval()

    return model


def merge_boxes(results_rois, image_size):
    # Ensure results_rois is a numpy array
    if isinstance(results_rois, torch.Tensor):
        results_rois = results_rois.cpu().detach().numpy()

    # Convert each ROI into a set of coordinates representing the corners of the box
    boxes = []
    for box in results_rois:
        ymin, xmin, ymax, xmax ,_,_= box
        coors = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.int32)
        boxes.append(coors)

    # Determine the size of the stencil based on the provided image size
    stencil_size = [image_size[1], image_size[0], 3]

    color = [255, 255, 255]

    for i in range(len(boxes)):
        stencil1 = np.zeros(stencil_size, dtype=np.uint8)
        contours = [boxes[i]]
        cv2.fillPoly(stencil1, contours, color)

        for j in range(i + 1, len(boxes)):
            stencil2 = np.zeros(stencil_size, dtype=np.uint8)
            contours = [boxes[j]]
            cv2.fillPoly(stencil2, contours, color)

            intersection = np.sum(np.logical_and(stencil1, stencil2))

            if intersection > 0:
                xmin = min(boxes[i][0][0], boxes[j][0][0])
                ymin = min(boxes[i][0][1], boxes[j][0][1])
                xmax = max(boxes[i][2][0], boxes[j][2][0])
                ymax = max(boxes[i][2][1], boxes[j][2][1])

                results_rois[i] = [ymin, xmin, ymax, xmax , -1,-1]
                results_rois = np.delete(results_rois, j, 0)

                return merge_boxes(results_rois, image_size)

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
    # Perform inference
    results = model(image)
    bboxes = results.xyxy[0]  # Bounding boxes (x1, y1, x2, y2, confidence, class)
    image_size = image.shape
    merged = merge_boxes(bboxes,image_size)
    extract_imgs = list()
    for bbox in merged:
        # print(bbox)
        # print(bbox[0])
        xmin, ymin, xmax, ymax ,_,_= bbox.astype(np.int16)

        #  cropped_img = img[y: y + h, x: x + w]
        cropped_img = image[ymin:ymax, xmin: xmax]
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
    print(FPS , total )
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

    model = whisper.load_model('base')
    result = model.transcribe(audio)

    return result['text']



def STT_summar(pre_dir , video_name):
    audio_output_path = make_Audio(pre_dir,video_name)
    #audio_dir = 'tmp_audio/'+video_name.split('.')[0]+'/' + video_name.split('.')[0]+'.wav'
    text = transcribe(audio_output_path)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    ratio = 0.3
    summary = summarizer(parser.document, int(len(text.split()) * ratio))
    return " ".join(str(sentence) for sentence in summary) , text 

    # stt_text = gensim.summarization.summarize(text,ratio=0.3)
    # return stt_text , text

def markdown(img_name,data,summary , stt ,imgCropped):
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
                        lst.append(str(t)+'\n')
            if t[1]==numCluster-2:
                lst.append(str(t[0])+'\n')
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
    mdFile.new_header(level=2,title = "Summary : ")
    mdFile.new_paragraph(text=summary)
    mdFile.new_line("\pagebreak") 
    mdFile.new_header(level=2,title = "STT : ")
    mdFile.new_paragraph(text=stt)
    
    mdFile.create_md_file()
    doc = aw.Document(f'./{file_name}.md')
    doc.save(f'./{file_name}.pdf')
    # output = pypandoc.convert_file('./'+file_name+'.md', 'pdf', outputfile='./'+file_name+".pdf")


def main_solution(model,pre_dir , file_name):


    imgDeleted,imgCropped,data = extract_from_video(model,pre_dir+file_name)
    

    summary, text = STT_summar(pre_dir,file_name)

    #stt = "testing / we are testing this system. Is this work?"
    markdown(file_name,data,summary , text ,imgCropped)