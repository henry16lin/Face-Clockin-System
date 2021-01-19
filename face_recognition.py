  
import glob
import os
import logging
import datetime
import numpy as np
import cv2
import time
from timeit import default_timer as timer
from PIL import Image,ImageDraw, ImageFont 

normalLogger = logging.getLogger('normalLogger')


def face_registed_projector(model):
    #run when have not registed to npy file
    img_list_jpg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpg'))
    img_list_jpeg = glob.glob(os.path.join(cwd,'img_for_registed','*.jpeg'))
    img_list_png = glob.glob(os.path.join(cwd,'img_for_registed','*.png'))
    img_list = img_list_jpg + img_list_jpeg + img_list_png
    fact_cnt = 0
    for file in img_list:
        #img = cv2.imread(file)
        img=cv2.imdecode(np.fromfile(file,dtype=np.uint8),-1)
        if '.jp' in file:
            file_name = os.path.basename(file).split('.jp')[0]
        elif '.png' in file:
            file_name = os.path.basename(file).split('.png')[0]
        elif '.JPG' in file:
            file_name = os.path.basename(file).split('.JPG')[0]
        print('registering for %s'%file_name)
        try:
            faces,points,bbox = model.get_input(img)
        except:
            print('fail to read %s, which will be ommitted'%file_name)      
        
        for i in range(len(faces)):
            face = faces[i]
            f1 = model.get_feature(face)
            #print('feature shape:')
            #print(f1.shape)
        
            margin = 44
            x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
            y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
            x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), img.shape[1]) )
            y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), img.shape[0]) )

            
            if i>=1:
                npy_name = file_name+'_'+str(i)+'.npy'
                jpg_name = file_name+'_'+str(i)+'.jpg'
            else:
                npy_name = file_name+'.npy'
                jpg_name = file_name+'.jpg'
            np.save(os.path.join(cwd,registed_folder,npy_name),f1)
            #cv2.imwrite(os.path.join(cwd,registed_folder,jpg_name), img[y1:y2, x1:x2])
            cv2.imencode('.jpg', img[y1:y2, x1:x2])[1].tofile(os.path.join(cwd,registed_folder,jpg_name))
            fact_cnt+=1
    print('successfully register %d images，total %d faces!'%(len(img_list),fact_cnt))


def registed_face_loader(registed_folder):
    ## load registed npy ##
    registed_npy_list = glob.glob(os.path.join(registed_folder, '*.npy'))
    registed_feature = []
    cat = []
    for npy in registed_npy_list:
        f1 = np.load(npy)
        cat.append(os.path.basename(npy).split('.npy')[0])
        registed_feature.append(f1)
    if registed_npy_list==[]:
        #print('there is no .npy file in registed face folder')
        normalLogger.debug('there is no .npy file in registed face folder')
    else:
        #print('load registed %d faces with %d dimensions'%(len(registed_feature),registed_feature[0].shape[0]))
        normalLogger.debug('load registed %d faces with %d dimensions'%(len(registed_feature),registed_feature[0].shape[0]))
    #print('registed names:')
    #print(cat)
    normalLogger.debug('registed names: '+str(cat))
    return registed_feature,cat



def face_comparison(args, img, registed_feature, cat, model, threshold):
    faces,points,bbox = model.get_input(img)
    
    #cv2 format to PIL format for 中文字label
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #cv2 to PIL
    
    all_cat,all_sim = [],[]
    people, people_sim = [],[]
    if (faces==[] or points==[] or bbox==[]):
        img = img  
    
    else:
        print('found %d faces'%faces.shape[0])
        text_size = np.floor(3e-2 * img_PIL.size[1]).astype('int32')
        #fontText = ImageFont.truetype("font/simsun.ttc", text_size, encoding="utf-8")
        fontText = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', text_size, encoding="utf-8")
        #fontText = ImageFont.truetype("/data/AI/user/henry/simsun.ttc", text_size, encoding="utf-8")
        thickness = (img_PIL.size[0] + img_PIL.size[1]) // 600
        
        now_time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        for i in range(faces.shape[0]):
            
            face = faces[i]
            #print(face)
            #print(face.shape)
            f2 = model.get_feature(face)
            if args.ga:
                gender, age = model.get_ga(face)
            
            sim_record = np.matmul(registed_feature,f2.T)
            most_sim_ind = np.argmax(sim_record)
            
                       
            margin = 10
            x1 = int(np.maximum(np.floor(bbox[i][0]-margin/2), 0) )
            y1 = int(np.maximum(np.floor(bbox[i][1]-margin/2), 0) )
            x2 = int(np.minimum(np.floor(bbox[i][2]+margin/2), img.shape[1]) )
            y2 = int(np.minimum(np.floor(bbox[i][3]+margin/2), img.shape[0]) )
            
            draw = ImageDraw.Draw(img_PIL)
            if sim_record[most_sim_ind]>=threshold:
                people.append(cat[most_sim_ind])
                people_sim.append(np.round(sim_record[most_sim_ind],3))

                if args.ga:
                    if gender == 1:
                        text = cat[most_sim_ind]+','+str(np.round(sim_record[most_sim_ind],3))+',Male,'+str(age)
                    else:
                        text = cat[most_sim_ind]+','+str(np.round(sim_record[most_sim_ind],3))+',Female,'+str(age)
                else:
                    text = cat[most_sim_ind]+','+str(np.round(sim_record[most_sim_ind],3))
                
                label_size = draw.textsize(text, fontText)
                if y1 - label_size[1] >= 0:
                    text_origin = np.array([x1, y1 - label_size[1]])
                else:
                    text_origin = np.array([x1, y1 + 1])
                
                for i in range(thickness):
                    draw.rectangle([x1+i,y2+i,x2-i,y1-i],outline="red")#畫框
                    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],outline="red",fill="red")#字體background的框
                    draw.text(text_origin, text, (255, 255, 255), font=fontText)
                
                #save found face
                #save_path = os.path.join(cwd,'found_face',cat[most_sim_ind])
                #if not os.path.exists(save_path):
                #    os.makedirs(save_path)
                #img_name = now_time +'_'+cat[most_sim_ind]+'_'+str(np.round(sim_record[most_sim_ind],3))+ '.jpg'
                #cv2.imencode('.jpg', img[y1:y2, x1:x2])[1].tofile(os.path.join(save_path,img_name))
                #cv2.imwrite(os.path.join(save_path,img_name),img[y1:y2, x1:x2])
                
            else:
                for i in range(thickness):
                    draw.rectangle([x1+i,y2+i,x2-i,y1-i],outline="green" )#畫框
                    if args.ga:
                        if gender==1:
                            text = 'Male,'+str(age)
                        else:
                            text = 'Female,'+str(age)
                        label_size = draw.textsize(text, fontText)
                        if y1 - label_size[1] >= 0:
                            text_origin = np.array([x1, y1 - label_size[1]])
                        else:
                            text_origin = np.array([x1, y1 + 1])
                        
                        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],outline="green",fill="green")#字體background的框
                        draw.text(text_origin, text, (255, 255, 255), font=fontText)
                    
            
            #tony_resize = tony.resize((x2-x1, y2-y1))
            #img_PIL.paste(tony_resize, (x1, y1))
            
            del draw
            
            all_cat.append(cat)
            all_sim.append(sim_record)
        
        
    return all_cat, all_sim, img_PIL, people, people_sim


def face_comparison_video(args, registed_feature,cat,model,input_path,output_path,cam_num, threshold):

    if input_path=='':
        vid = cv2.VideoCapture(cam_num)
    else:
        vid = cv2.VideoCapture(input_path)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    #video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    start_time = time.time()
    while True:
        return_value, frame = vid.read()
        
        _, _, image, people, people_sim = face_comparison(args, frame, registed_feature, cat, model, threshold)

        img_OpenCV = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) #轉回cv2 format 否則RGB亂掉
        result = np.asarray(img_OpenCV)


        time_out_flag = (time.time()-start_time)>30 # if timeout, then return people=[]
        if len(people)>0 or time_out_flag: #if detected some person, then return result image and break while loop
            cv2.destroyAllWindows()
            for i in range (1,5):
                cv2.waitKey(1)
            break
            
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        
        print(fps)

        if isOutput:
            out.write(result)

        if not args.no_show:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return people, people_sim, result






        