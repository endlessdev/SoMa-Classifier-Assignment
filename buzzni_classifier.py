# author: Jade Yeom <ysw0094@gmail.com>
# date: 17/10/18
# description classifier for products

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from konlpy.tag import Kkma
from sklearn.model_selection import GridSearchCV

import requests

import logging.handlers

import sys
import numpy as np
import json

logger = logging.getLogger('classifier')
formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.setLevel(logging.DEBUG)

kkma = Kkma()

def titleToNounsText (title):
    return " ".join(kkma.nouns(title))

x_text_list = []
y_text_list = []

enc = sys.getdefaultencoding()

with open("refined_category_dataset.dat", encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        edited_title = titleToNounsText(info['name'])
        logger.info(info['pid'] +' '+ info['name'])
        x_text_list.append((info['pid'], edited_title))
        y_text_list.append(info['cate'])

y_name_id_dict = joblib.load("y_name_id_dict.dat")

logger.info(y_name_id_dict)

y_id_list = [y_name_id_dict[x] for x in y_text_list]

vectorizer = TfidfVectorizer()
x_list = vectorizer.fit_transform(map(lambda i: i[1], x_text_list))
y_list = [y_name_id_dict[x] for x in y_text_list]

logger.info(x_list)
logger.info(y_list)

logger.info(x_text_list[2])

X_train, X_test, y_train, y_test = train_test_split(x_text_list, y_list, test_size=0.2, random_state=42)

for c in [1, 5, 10]:
    clf = LinearSVC(C = c)
    X_train_text = map(lambda i: i[1], X_train)
    clf.fit(vectorizer.transform(X_train_text), y_train)
    print(c, clf.score(vectorizer.transform(map(lambda i: i[1], X_test)), y_test))

svc_param = np.logspace(-1, 1, 4)
gscv = GridSearchCV(LinearSVC(), param_grid={'C': svc_param}, cv=10, n_jobs=3)
gscv.fit(vectorizer.transform(map(lambda i: i[1], x_text_list)), y_list)

logger.info(gscv.best_score_, gscv.best_params_)

eval_x_text_list = []

with open("soma8_eval_data.dat", encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        eval_x_text_list.append((info['pid'], titleToNounsText(info['name'])))

pred_list = clf.predict(vectorizer.transform(map(lambda i: i[1], eval_x_text_list)))

evalParam = {
    'name': '염승우',
    'nickname': 'lokihardt',
    'mode': 'eval',
    'pred_list': ",".join(map(lambda i: str(int(i)), pred_list.tolist()))
}

evalRequest = requests.post('http://eval.buzzni.net:20001/eval', data=evalParam)
logger.info(evalRequest.json())


# ### CNN 으로 추출한 이미지 데이터 사용하기
#  * keras mobilenet 으로 추출한 데이터, 이 데이터를 아래처럼 읽어서 사용 가능함
#  * 더 성능이 높은 모델로 이미지 피쳐를 추출하면 성능 향상 가능함
# In[20]:
# pid_img_feature_dict = {}
# with open("refined_category_dataset.img_feature.dat") as fin:
#     for idx,line in enumerate(fin):
#         if idx % 100 == 0:
#             print(idx)
#         pid, img_feature_str = line.strip().split(" ")
#         img_feature = (np.asarray(list(map(lambda i : float(i),img_feature_str.split(",")))))
#         pid_img_feature_dict[pid] = img_feature
# #         print (line)
# #         break
#
#
#
# # In[21]:
#
#
# from scipy import sparse
#
#
# # In[59]:
#
#
# img_feature_list = []
# for pid, name in X_train:
# #     print(pid, name)
#     if pid in pid_img_feature_dict:
#         img_feature_list.append(pid_img_feature_dict[pid])
# #         print (len(pid_img_feature_dict[pid]),type(pid_img_feature_dict[pid]))
# #         break
#     else:
#         img_feature_list.append(np.zeros(1000))
# #     break
#
#
# # In[65]:
#
#
# img_feature_test_list = []
# for pid, name in X_test:
#     if pid in pid_img_feature_dict:
#         img_feature_test_list.append(pid_img_feature_dict[pid])
#     else:
#         img_feature_test_list.append(np.zeros(1000))
#
#
# # In[61]:
#
#
# print(len(img_feature_list))
#
#
# # #### 아래 부분은 text feature 와 이미지 feature 를 합쳐서 feature 를 만드는 부분이다. 이 부분에 대해서는 각자 한번 합치는 방법을 찾아 보면 된다.
#
# # In[63]:
#
# # concat_x_list = vectorizer.transform(map(lambda i : i[1],X_train)), img_feature_list
# # concat_test_x_list = (vectorizer.transform(map(lambda i : i[1],X_test)), img_feature_test_list)
#
#
# concat_x_list = sparse.hstack((vectorizer.transform(map(lambda i : i[1],X_train)), img_feature_list))
# concat_test_x_list = sparse.hstack((vectorizer.transform(map(lambda i : i[1],X_test)), img_feature_test_list))
# # In[103]:
#
#
# for c in [1]:
#     clf2 = LinearSVC(C=c)
#     clf2.fit(concat_x_list, y_train)
#     print (c,clf2.score(concat_test_x_list, y_test))
#
# # In[89]:
#
#
# del pid_img_feature_dict
#
#
# # ### CNN 피쳐를 추가 해서 분류후 평가 서버에 분류 결과를 전송
#
# # In[90]:
#
#
# pid_img_feature_dict = {}
# with open("refined_category_dataset.img_feature.dat") as fin:
#     for idx,line in enumerate(fin):
#         if idx%100 == 0:
#             print(idx)
#         pid, img_feature_str = line.strip().split(" ")
#         img_feature = (np.asarray(list(map(lambda i : float(i),img_feature_str.split(",")))))
#         pid_img_feature_dict[pid] = img_feature
# #         print (line)
# #         break
#
#
#
# # In[99]:
#
#
# test_x_text_list = []
# with open("soma8_test_data.dat",encoding=enc) as fin:
#     for line in fin.readlines():
#         info = json.loads(line.strip())
#         test_x_text_list.append((info['pid'],info['name']))
#
# # pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],eval_x_text_list)))
#
#
# # In[100]:
#
#
# img_feature_eval_list = []
#
#
# # In[101]:
#
#
# for pid, name in test_x_text_list:
#     if pid in pid_img_feature_dict:
#         img_feature_eval_list.append(pid_img_feature_dict[pid])
#     else:
#         img_feature_eval_list.append(np.zeros(1000))
#
#
# # In[102]:
#
#
# print (len(img_feature_eval_list), len(eval_x_text_list))
#
#
# # In[103]:
#
#
# x_feature_list = vectorizer.transform(map(lambda i : i[1],test_x_text_list))
#
#
# # #### 2개 feature 를 합치는 방법 찾아보기
#
# # In[104]:
#
#
# concat_test_x_list = sparse.hstack((x_feature_list, img_feature_eval_list),format='csr')
#
#
# # In[105]:
#
#
# pred_list = clf2.predict(concat_test_x_list)
# evalParam = {
#     'name': '염승우',
#     'nickname': 'lokihardt',
#     'mode': 'test',
#     'pred_list': ",".join(map(lambda i: str(int(i)), pred_list.tolist()))
# }
#
# evalRequest = requests.post('http://eval.buzzni.net:20001/eval', data=evalParam)
# logger.info(evalRequest.json())