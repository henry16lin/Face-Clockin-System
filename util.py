import pandas as pd
import datetime
import sqlite3
import os

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders



def folder_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)


def table_creator(cursor):
    try:
        insert_sql_str = """CREATE TABLE people_check_dtl (
            DATE          TEXT NOT NULL,
            PERSON        TEXT NOT NULL,
            CHECKINTIME   TEXT,
            CHECKOUTTIME  TEXT,
            RPT_DATETIME    TEXT NOT NULL,
            PRIMARY KEY(DATE,PERSON)
        );"""
        cursor.execute(insert_sql_str)
    except:
        print('table already exist, parse data and insert to exists table...')



def daily_summary():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    print(today)
    sql_str = ' select * from people_check_dtl where DATE = "%s";' %today
    print(sql_str)
    conn = sqlite3.connect("test.db")
    daily_summary_pd = pd.read_sql_query(sql_str, conn)
    conn.commit()
    conn.close

    csv_path = r'daily_summary/'+ today + 'summary.csv'
    daily_summary_pd.to_csv( csv_path, index=False)

    print('send summary email......')
    email_senter(today, csv_path)


def email_senter(date, attach_path):
    try:
        subject = date +' 簽到總表'
        content = 'FYI'
        csv_path = r'daily_summary/' + date +'Summary.csv'
        send_mail(subject,content,attach_path_list=[csv_path])
        print('successfully send email!')
    except:
        print('fail to send email. ignore it!')



def send_mail(subject, content, attach_path_list=[]):
    email_user = 'your@mail.com'
    email_send = ['receiver@mail.com']
    
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = ', '.join(email_send)
    msg['Subject'] = subject
    
    body = content
    msg.attach(MIMEText(body,'plain'))
    
    for i in range(len(attach_path_list)):
        part = MIMEBase('application','octet-stream')
        filename = attach_path_list[i]
        attachment = open(filename,'rb')
        part = MIMEBase('application','octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',"attachment; filename= "+filename)
        msg.attach(part)


    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login("your@mail.com", "yourpassword")
    
    server.sendmail(email_user,email_send,text)
    server.quit()

    
    return True

