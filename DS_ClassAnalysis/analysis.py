#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import sys
import time
import os
from PIL import Image, ImageFont, ImageDraw
from collections import Counter
import shutil


plt.rcParams['font.family'] = ['Adobe Fangsong Std']

teachWhat = []
classComment = []
commentWord = []
timeArray = []
title_list = []
answer_list = []
Answer = np.zeros((10,6))

# 读取csv至字典
with open('classcomment.csv') as file:
    file_csv = csv.reader(file)
    headers = next(file_csv)
    for row in file_csv:
        word = row[-1]
        year = word.rfind("2019")
        timeNow = word[year:year+10]
        timeArray.append(timeNow)
        word = word[0:word.find("]")]
        if(len(word) >= 12):
            word = word.replace("\\n","  ")
            word = word[0:len(word)-2]
            commentWord.append(word)
        temp = row[0]
        temp = temp[temp.rfind("[") + 1:]
        temp[temp.rfind("[") + 1:]
        classComment.append(row[1:11])
        teachWhat.append(temp[2:len(temp)-2])

when = Counter(timeArray).most_common(1)[0][0]
teachContent = Counter(teachWhat).most_common(1)[0][0]
for index,char in enumerate(teachContent):
    if(char.find("/") != -1):
        teachContent = teachContent[0:index] + "|" + teachContent[index+1:]
teachContent = "(" + teachContent +")"


Question = [
      {
        "names": "你觉得这节课整体难度如何？",
        "choice": ["好难诶", "还好啦", "较容易"],
      },
      {
        "names": "你觉得自己这节课的精神状态如何？",
        "choice": ["可以说是一直全神贯注了", "有时想睡有时认真听", "真的超级超级想睡觉"],
      },
      {
        "names": "你觉得这节课的趣味性如何？",
        "choice": ["啊啊啊超好玩", "一般一般一般", "呜呜有点无聊"],
      },
      {
        "names": "你对这节课所涉及的知识点是否搞明白了呀?",
        "choice": ["全部搞懂啦", "还好还需课后复习", "感觉这节课白上了"],
      },
      {
        "names": "你对这节课所涉及代码是否搞明白了?",
        "choice": ["我好聪明全部搞懂啦", "还好还好还需课后复习", "呜呜呜简直天书一般"],
      },
      {
        "names": "这节课你最大的收获是?",
        "choice": ["数据结构的概念", "数据结构的具体操作", "代码复现", "呜呜呜呜没有"],
      },
      {
        "names": "最吸引你来上课的动力是?",
        "choice": ["3.5个学分", "学起来超好玩", "数据结构很重要", "老师的魅力!!", "其他原因"],
      },
      {
        "names": "请问这节课你坐在第几排",
        "choice": ["前 3 排", "中间几排", "后 3 排"],
      },
      {
        "names": "你对这节课的整体评星为?",
        "choice": ["1 星", "2 星 ", "3 星", "4 星","5 星"],
      },
    ]

for item in Question :
    title_list.append(item["names"])
    answer_list.append(item["choice"])

for comment in classComment:
    for ID,piece in enumerate(comment):
        if(piece.isdecimal()):
            piece = int(piece)
            Answer[ID,piece] += 1

#coding=utf-8
for i in range(9):
    plt.title(title_list[i])
    plt.bar(range(len(answer_list[i])),Answer[i,0:len(answer_list[i])],color='rgb',
            tick_label=answer_list[i],width=0.4,lw=5,alpha=0.4)
    for x,y in zip(range(len(answer_list[i])),Answer[i,0:len(answer_list[i])]):
        plt.text(x, y + 0.05, '%d' % y, ha='center', va='bottom')
    fig = plt.gcf()
    plt.savefig("tempImage/" + when + teachContent + title_list[i],dpi=400)
    plt.close(fig)    

fileName = "(" + when + ")" + "Comments" + ".txt"
f = open(fileName,"a+")
f.write(when + "\n\n" + teachContent + "\n\n")
for index, word in enumerate(commentWord):
    text = str(index) + " --> " + word + "\n\n\n"
    f.write(text)

images = [Image.open("tempImage/" + image) for image in os.listdir("tempImage/") if image.endswith(".png")]
width,height = images[0].size
wordImage = Image.new('RGB', (width, int(height*len(commentWord)/23)), color=(255,255,255))
draw = ImageDraw.Draw(wordImage)
font = ImageFont.truetype("AdobeHeitiStd-Regular.otf",size=30)
colorarrow = 'rgb(0, 0, 0)'
colorword = 'rgb(255,86,156)'
for index, word in enumerate(commentWord):
    textarrow = str(index) + " --> "
    (x, y) = (20, index * 50 +20)
    draw.text((x, y), textarrow, fill=colorarrow, font=font)
    (x, y) = (110, index * 50 +20)
    draw.text((x, y), word, fill=colorword, font=font)

result = Image.new('RGB', (width,  height*len(images) + int(height*len(commentWord)/23)) )
last = len(images)
for i, image in enumerate(images):
    result.paste(image, box = (0, i*height)) 
result.paste(wordImage, box = (0, len(images)*height))
resultName = "(" + when + ")" + "Result.jpg"
result.save(resultName)


oriName = "classcomment.csv"
newName = "(" + when + ")" + teachContent + "[Comment].csv"

os.rename(oriName,  "(" + when + ")" + teachContent + "[Comment].csv" )
shutil.move(src=newName,dst="classCommentData/")

shutil.copyfile(src=resultName,dst="finalResult/" + resultName)
shutil.move(src=fileName,dst="finalResult/")