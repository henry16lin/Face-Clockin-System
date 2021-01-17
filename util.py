import pandas as pd

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


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


'''
def db_update_checker():
    # whether db has update
    return True/False


def summary_daily_result:
    # access sqlite and query daily result
    return pd_dataframe

'''


def send_mail(subject, content, attach_path_list=[]):
    email_user = 'jiweilin0907@gmail.com'
    email_send = ['peaceful0907@gmail.com']
    
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
    server.login("your@email.com", "password")
    
    server.sendmail(email_user,email_send,text)
    server.quit()

    
    return True

