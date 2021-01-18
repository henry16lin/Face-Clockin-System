# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, send_file,abort, url_for, redirect

import uuid
import os
import logging
import datetime
import time
import threading
import argparse
from glob import glob

import cv2
import sqlite3
import pandas as pd

import face_model
import face_recognition

from util import folder_checker, table_creator, daily_summary


app = Flask(__name__)

cwd = os.getcwd()
normalLogger = logging.getLogger('normalLogger')


os.environ['MXNET_SUBGRAPH_VERBOSE'] = "0"


@app.route('/')
def test():
    return "<h1>This is a test page!</h1>"


@app.route('/home', methods=['GET', 'POST'])
def home():

    if request.method == 'POST' :
        submit_type = request.form['submit_button']
        
        uid = str(uuid.uuid1())
        now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H%:%M:%S")

        return render_template('loading.html',
                               loading_img='./static/loading.gif',
                               uid = uid, submit_type = submit_type, time_str = now_time_str)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template('home.html',datetime=time_str)
 


@app.route('/recognition/<time_str>/<uid>/<submit_type>')
def result(time_str, uid, submit_type):
    #open cam and do face recognition

    people, people_sim, result_image_cv2 = face_recognition.face_comparison_video(args, registed_feature, cat, model, 
                                                                input_path='', output_path='', cam_num=0, threshold=0.4)
    
    if len(people) == 0: #time out, back to home page
        return redirect(url_for('home'))

    save_path = os.path.join(cwd, 'static','images')
    img_name = uid + '.jpg'
    cv2.imencode('.jpg', result_image_cv2)[1].tofile(os.path.join(save_path, img_name))
    
    date = time_str.split(' ')[0]

    conn = sqlite3.connect('test.db') #create/connect to db

    all_note = ''
    for person in people: # 多人簽到！！
        if submit_type.find('簽到')!=-1:
            note = clockin(conn, date, person, time_str)

        elif submit_type.find('簽退')!=-1:
            note = clockout(conn, date, person, time_str)
        
        all_note = all_note + '\n' +note

    conn.commit()
    conn.close
    
    return render_template("result.html", 
                           datetime = time_str,
                           image_path='../../../../../../static/images/%s.jpg'%uid,
                           remark = all_note
                           )


def clockin(conn, date, person, time_str):
    #check whether have clockin. if not then insert db  else skip

    sqlstr = 'select * from people_check_dtl where DATE = "%s" and PERSON = "%s" and CHECKINTIME is not NULL;'%(date, person)
    query_result_pd = pd.read_sql_query(sqlstr, conn)

    if len(query_result_pd)==0: # this person has not clockin yet

        rpt_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sqlstr = 'INSERT OR REPLACE into people_check_dtl (DATE,PERSON,CHECKINTIME,RPT_DATETIME) values(?,?,?,?)'
        cursor = conn.cursor()
        cursor.execute(sqlstr, [date, person, time_str, rpt_datetime])

        note = '%s clockin successfully on %s!'%(person,time_str)

    else:
        sqlstr = 'select CHECKINTIME from people_check_dtl where DATE = "%s" and PERSON = "%s";'%(date, person)
        query_result_pd = pd.read_sql_query(sqlstr, conn)
        first_checkin_time = query_result_pd['CHECKINTIME'][0]
        note = '%s already clockin on %s, do not do anything...' %(person,first_checkin_time)
    
    return note


def clockout(conn, date, person, time_str):
    sqlstr = 'select CHECKINTIME from people_check_dtl where DATE = "%s" and PERSON = "%s" and CHECKINTIME is not NULL;'%(date, person)
    query_result_pd = pd.read_sql_query(sqlstr, conn)
    if len(query_result_pd)>0:
        checkintime = query_result_pd['CHECKINTIME'][0]
    else:
        checkintime = None

    #keep recover clockout time
    rpt_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    sqlstr = 'INSERT OR REPLACE into people_check_dtl (DATE,PERSON,CHECKINTIME,CHECKOUTTIME,RPT_DATETIME) values(?,?,?,?,?)'
    cursor = conn.cursor()
    cursor.execute(sqlstr, [date, person, checkintime, time_str, rpt_datetime])

    note = '%s clockout out on: %s'%(person,time_str)
    return note




def SetupLogger(loggerName, filename):
    path = os.path.join(cwd,'log')
    if not os.path.exists(path):
        os.makedirs(path)

    logger = logging.getLogger(loggerName)

    logfilename = datetime.datetime.now().strftime(filename)
    logfilename = os.path.join(path, logfilename)

    logformatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(logfilename, 'a', 'utf-8')
    fileHandler.setFormatter(logformatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logformatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    


def old_face_cleaner():
    while True:
        for f in glob(os.path.join(cwd,'static','images','*.jpg')):
            mtime = os.path.getmtime(f)
            mtime_ = datetime.datetime.fromtimestamp(mtime)
            now_time = datetime.datetime.now()
            if mtime_< (now_time- datetime.timedelta(minutes=60)):
                normalLogger.debug('delete old graph:%s'%f)
                os.remove(f)
        time.sleep(3600)


def time_checker(email_time):
    while True:
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for t in email_time:
            if time_str.find(t)!=-1:
                print(t, 'time to sent email!')
                daily_summary()

        time.sleep(59)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face clockin system')
    parser.add_argument('--image-size', default='112,112', help='') #follow sphere face & cosin face format
    parser.add_argument('--model', default='./models/mobilefacenet/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='./gamodel-r50/model,0', help='path to load model.')
    parser.add_argument('--ga',default=False,action="store_true",help='whether to estimate gender and age')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id,(-1) for CPU')
    parser.add_argument('--cam', default=0, type=int, help='which cam used')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument('--image_mode', default=False, action="store_true",help='Image detection mode')
    parser.add_argument('--regist', default=False, action="store_true",help='to regist face or to compare face')
    parser.add_argument('--no_show', default=False, action="store_true",help='to regist face or to compare face')
    parser.add_argument('--face_threshold', default=0.4, type=float, help='similarity threshold')
    args = parser.parse_args()
    
    SetupLogger('normalLogger', "%Y-%m-%d.log")
    
    #load model
    model = face_model.FaceModel(args)
    cwd = os.getcwd()
    
    if 'mobileface' in args.model:
        registed_folder = os.path.join(cwd, 'registed_img_mobile')
    else:
        registed_folder = os.path.join(cwd, 'registed_img_r100')

    
    email_time = ['12:00', '16:40']

    # sub-tread check old image
    cleaner = threading.Thread(target = old_face_cleaner, daemon=True)
    emailer = threading.Thread(target = time_checker, args=(email_time,), daemon=True)
    
    cleaner.start()
    emailer.start()
    
    registed_feature, cat = face_recognition.registed_face_loader(registed_folder)

    '''
    # table checke
    conn = sqlite3.connect('test.db') #create/connect to db
    cursor = conn.cursor()
    table_creator(cursor)    #create table if table doesn't exist
    conn.commit()
    conn.close
    '''
    
    app.run(
        debug = False,
        host = '0.0.0.0',
        port = 5566,
        threaded = False
    )
    
    #daily_summary()
    



