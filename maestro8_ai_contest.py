
# coding: utf-8

# # 8기 과제 - 딥러닝 기반 상품 카테고리 자동 분류 서버

# ## 과제 개요
# * 출제자 : 남상협 멘토 (justin@buzzni.com) / 버즈니 (http://buzzni.com) 대표
# * 배경 : 카테고리 분류 엔진은 실제로 많은 서비스에서 사용되는 중요한 기계학습 기술이다. 본 과제의 주제는 버즈니 개발 인턴이자 마에스트로 6기 멘티가 아래와 나와 있는 기본 분류 모델을 기반으로 deep learning 기반의 feature 를 더해서 고도화된 분류 엔진을 만들어서 2016 한국 정보과학회 논문으로도 제출 했던 주제이다. 기계학습에 대한 학습과, 실용성 두가지 측면에서 모두 도움이 될 것으로 보인다.
# 

# ## 과제 목표
# * 입력 : 상품명, 상품 이미지
# * 출력 : 카테고리
# * 목표 : 가장 높은 정확도로 분류를 하는 분류 엔진을 개발
# 

# 
# ## 평가 항목 
# * 성능평가 (100%)
#  
# ## 제출 항목 
# * 채점 서버에 자신이 분류한 class id 리스트를 파라미터로 넣어서 호출한다. 
# * name - 자신의 이름을 넣는다. 실제 점수판에는 공개가 안됨, 추후 평가시에 일치하는 이름의 멘티 점수로 사용함. 요청한 평가 중에서 가장 높은 점수의 평가 점수로 업데이트됨.
# * nickname - 점수판에 공개되는 이름, 자신의 이름으로 해도 되고, 닉네임으로 해도 됨. 구분을 위해서 사용하는 feature(text, textimage) 와 알고리즘 (svm, cnn) 등을 닉네임 뒤에 붙여준다. 
# * pred_list - 분류한 카테고리 id 리스트를 , 로 묶은 데이터 
# * 평가 점수가 반환된다. - precision, 높을 수록 좋다. 두가지 방법 각각 50%씩 점수 반영 
# * mode - 'test' 로 호출하면 웹으로 순위가 공개되는 테스트 평가를 수행하고 결과 점수가 반환된다. 해당 결과 점수는 http://eval.buzzni.net:20002/score 에서 확인 가능함. 실제 성적 평가는 'eval' 로 평가용 데이터로 호출하면 된다. 이때는 점수가 반환되거나, 웹 점수 보드에도 나오지 않는다. 
# * 너무 자주 평가를 요청하기 보다, 가급적 자체적으로 평가 해서, 괜찮게 나올때 요청하길 권장 
# ```python
# import requests
# name='test1'
# nickname='test1_text_svm'
# mode='test' #'eval' 을 실제 성적 평가용. 분류 점수 반환 안됨.
# param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list)),
#          'name':name,'nickname':nickname,"mode":mode}
# d = requests.post('http://eval.buzzni.net:20001/eval',data=param)
# print (d.json())         
#          ```

# ## 성능 향상 포인트
# * http://localhost:8000/notebooks/maestro8_deeplearning_product_classifier.ipynb 이 노트북에 있는 딥러닝 기반의 분류기로 분류할 경우에 더 높은 성능을 낼 수 있어서 유리함
# * 아래의 방법들은 하나의 예이고, 아래에 나와 있지 않은 다양한 방법들도 가능함.
# * 전처리 
#  * 오픈된 형태소 분석기(예 - konlpy) 를 써서, 단어 띄어쓰기를 의미 단위로 띄어서 학습하기
#  * bigram, unigram, trigram 등 단어 feature 를 더 다양하게 추가하기
# * 딥러닝 
#  * embedding weight 를 random 이 아닌 학습된 값을 사용하기 (https://radimrehurek.com/gensim/models/word2vec.html)
#  * 이미지 feature 를 CNN으로 추출할때 더 성능이 좋은 모델 사용하기 (예제로 준 데이터는 mobilenet 으로 성능보다 속도 위주로 된 모델)
#  * 다양한 파라미터(hyper parameter) 로 실험 해보기 
# * 피쳐 조합  
#  * 이미지 feature 와 text feature 를 합치는 부분 잘하기 
# 

# ## 평가 점수 서버 
# * 현재 평가 순위를 json 형태로 반환한다.
# * 여러번 호출했을때는 가장 높은 점수로 업데이트 한다.
#  * http://eval.buzzni.net:20002/score
# * 실제 점수는 

# In[2]:


import sys
from sklearn.externals import  joblib
from sklearn.grid_search import GridSearchCV
from sklearn.svm import  LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  TfidfVectorizer,CountVectorizer
import os
import numpy as np
import string
from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model



# ### 파일에서 학습 데이터를 읽는다.

# In[3]:


import json


# In[29]:


x_text_list = []
y_text_list = []
enc = sys.getdefaultencoding()
with open("refined_category_dataset.dat",encoding=enc) as fin:
    for line in fin.readlines():
#         print (line)
        info = json.loads(line.strip())
        x_text_list.append((info['pid'],info['name']))
        y_text_list.append(info['cate'])
        


# In[30]:


# joblib.dump(y_name_id_dict,"y_name_id_dict.dat")


# ### text 형식으로 되어 있는 카테고리 명을 숫자 id 형태로 변환한다.

# In[31]:


y_name_id_dict = joblib.load("y_name_id_dict.dat")


# In[32]:


print(y_name_id_dict)


# In[33]:



# y_name_set = set(y_text_list)
# y_name_id_dict = dict(zip(y_name_set, range(len(y_name_set))))
# print(y_name_id_dict.items())
# y_id_name_dict = dict(zip(range(len(y_name_set)),y_name_set))
y_id_list = [y_name_id_dict[x] for x in y_text_list]


# ### text 형태로 되어 있는 상품명을 각 단어 id 형태로 변환한다.

# In[71]:


from sklearn.model_selection import train_test_split

vectorizer = CountVectorizer()
x_list = vectorizer.fit_transform(map(lambda i : i[1],x_text_list))
y_list = [y_name_id_dict[x] for x in y_text_list]

print (x_list)
print (y_list)


# ### train test 분리하는 방법 

# In[72]:


print(x_text_list[2])


# In[73]:


X_train, X_test , y_train, y_test = train_test_split(x_text_list, y_list, test_size=0.2, random_state=42)


# In[45]:


# print(vectorizer.transform(list(map(lambda i : i[1],X_train))))


# ### 몇개의 파라미터로 간단히 테스트 하는 방법

# In[74]:



for c in [1,5,10]:
    clf = LinearSVC(C=c)
    X_train_text = map(lambda i : i[1],X_train)
    clf.fit(vectorizer.transform(X_train_text), y_train)
    print (c,clf.score(vectorizer.transform(map(lambda i : i[1],X_test)), y_test))


# ### 최적의 파라미터를 알아서 다 해보고, n-fold cross validation까지 해주는 방법 - GridSearchCV

# In[48]:


svc_param = np.logspace(-1,1,4)
# svc_param = np.logspace(2.0, 3.0, num=4)



# In[49]:



gsvc = GridSearchCV(LinearSVC(), param_grid= {'C': svc_param}, cv = 10, n_jobs = 4)


# In[50]:


gsvc.fit(vectorizer.transform(map(lambda i : i[1],x_text_list)), y_list)


# In[51]:


print(gsvc.best_score_, gsvc.best_params_)


# #### 평가 데이터에 대해서 분류를 한 후에  평가 서버에 분류 결과 전송

# In[53]:


eval_x_text_list = []
with open("soma8_test_data.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        eval_x_text_list.append((info['pid'],info['name']))


# In[59]:


pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],eval_x_text_list)))


# In[60]:


# print (pred_list.tolist())


# In[68]:


import requests
import datetime
name = "염승우"
nickname = 'lokihardt' + str(datetime.datetime.now().time())
print(nickname + "으로 두번째 리퀘스트")
mode = 'test'
param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
         'name':name,'nickname':nickname,'mode':mode}
d = requests.post('http://eval.buzzni.net:20001/eval',data=param)

print("test")
print (d.json())


# #### eval 데이터에 대해서 분류를 한 후에  평가 서버에 분류 결과 전송
#  * 실제 여기서 나온 점수로 채점을 한다.

# In[69]:


eval_x_text_list = []
with open("soma8_test_data.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        eval_x_text_list.append((info['pid'],info['name']))
pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],eval_x_text_list)))
nickname = 'lokihardt' + str(datetime.datetime.now().time())
print(nickname + "으로 두번째 리퀘스트")
mode = 'test'
param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
         'name':name,'nickname':nickname,'mode':mode}
d = requests.post('http://eval.buzzni.net:20001/eval',data=param)

print("eval")
print (d.json())

#
# # ### CNN 으로 추출한 이미지 데이터 사용하기
# #  * keras mobilenet 으로 추출한 데이터, 이 데이터를 아래처럼 읽어서 사용 가능함
# #  * 더 성능이 높은 모델로 이미지 피쳐를 추출하면 성능 향상 가능함
#
# # In[20]:
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
# concat_x_list = vectorizer.transform(map(lambda i : i[1],X_train)), img_feature_list
# concat_test_x_list = (vectorizer.transform(map(lambda i : i[1],X_test)), img_feature_test_list)
#
#
# # In[103]:
#
#
#
#
# for c in [1]:
#     clf2 = LinearSVC(C=c)
#     clf2.fit(concat_x_list, y_train)
#     print (c,clf2.score(concat_test_x_list, y_test))
#
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
# with open("refined_category_dataset.img_feature.eval.dat") as fin:
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
# with open("soma8_eval_data.dat",encoding=enc) as fin:
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
#
#
# # In[106]:
#
#
#
# import requests
# name = '염승우'
# nickname = 'lokihardt'
# mode = 'eval'
# param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
#          'name':name,'nickname':nickname,'mode':mode}
# d = requests.post('http://eval.buzzni.net:20001/eval',data=param)
# print (d.json())
#
#
#
# # In[107]:
#
#
# eval_x_text_list = []
# with open("soma8_eval_data.dat",encoding=enc) as fin:
#     for line in fin.readlines():
#         info = json.loads(line.strip())
#         eval_x_text_list.append((info['pid'],info['name']))
#
# # pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],eval_x_text_list)))
#
#
# # In[108]:
#
#
# img_feature_eval_list = []
# for pid, name in eval_x_text_list:
#     if pid in pid_img_feature_dict:
#         img_feature_eval_list.append(pid_img_feature_dict[pid])
#     else:
#         img_feature_eval_list.append(np.zeros(1000))
#
#
# # In[109]:
#
#
# x_feature_list = vectorizer.transform(map(lambda i : i[1],eval_x_text_list))
#
#
# # In[110]:
#
#
# concat_eval_x_list = sparse.hstack((x_feature_list, img_feature_eval_list),format='csr')
#
#
# # In[113]:
#
#
# pred_list = clf2.predict(concat_eval_x_list)
#
#
# # In[116]:
#
#
#
# import requests
# name = '염승우'
# nickname = 'lokihardt'
# mode = 'eval'
# param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
#          'name':name,'nickname':nickname,'mode':mode}
# d = requests.post('http://eval.buzzni.net:20001/eval',data=param)
# print (d.json())


