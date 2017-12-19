# coding: utf-8

# # 딥러닝 기반 상품 카테고리 자동 분류 서버 예
# ### 파일에서 학습 데이터를 읽는다.


import json
import sys
import keras
import keras.preprocessing.text
import numpy
import numpy as np
import theano
import requests
import os
import konlpy

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from konlpy.tag import Kkma, Mecab

from numpy import argmax
from keras.utils import to_categorical
hannanum = Mecab()

x_text_list = []
y_text_list = []
enc = sys.getdefaultencoding()
with open("refined_category_dataset.dat", encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        x_text_list.append((info['pid'], " ".join(hannanum.nouns(info['name']))))
        y_text_list.append(info['cate'])

# joblib.dump(y_name_id_dict,"y_name_id_dict.dat")
# ### text 형식으로 되어 있는 카테고리 명을 숫자 id 형태로 변환한다.

y_name_id_dict = joblib.load("y_name_id_dict.dat")

print(y_name_id_dict)


# y_name_set = set(y_text_list)
# y_name_id_dict = dict(zip(y_name_set, range(len(y_name_set))))
# print(y_name_id_dict.items())
# y_id_name_dict = dict(zip(range(len(y_name_set)),y_name_set))
y_list = [y_name_id_dict[x] for x in y_text_list]

# ### train test 분리하는 방법
X_train, X_test, y_train, y_test = train_test_split(x_text_list, y_list, test_size=0.2, random_state=42)

# ## 딥러닝 기반 text 분류에 필요한 모듈 로드

# In[9]:


# #### 모델 파일을 만약 만들었다면, 아래와 같이 로드 가능하다.

# In[10]:


# word2vec = gensim.models.KeyedVectors.load_word2vec_format('/workspace/resources/11st_all_product_name.segment.0918.15w100e3min.model', binary=True)
# word2vec.init_sims(replace=True)


# ### text 데이터를 word-id 형태로 변환한다.

# In[11]:


print(X_train)

sequence_tokenizer = keras.preprocessing.text.Tokenizer()
sequence_tokenizer.fit_on_texts(map(lambda i : i[1], X_train))
max_features = len(sequence_tokenizer.word_index)


def texts_to_sequences2(d_list, tokenizer, maxlen=300):
    seq = tokenizer.texts_to_sequences(d_list)
    print('mean:', numpy.mean([len(x) for x in seq]))
    print('std:', numpy.std([len(x) for x in seq]))
    print('median:', numpy.median([len(x) for x in seq]))
    print('max:', numpy.max([len(x) for x in seq]))
    seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    return seq


# In[ ]:




# In[ ]:


train = texts_to_sequences2(map(lambda i: i[1], X_train), sequence_tokenizer)
test = texts_to_sequences2(map(lambda i: i[1], X_test), sequence_tokenizer)

# #### word의 embedding 형태의 weight 를 초기화 한다.

# In[ ]:


input_dim = train.shape[1]

input_tensor = keras.layers.Input(shape=(input_dim,), dtype='int32')

# In[ ]:

word_vec_dim = 100
not_ct = 0
weights = numpy.zeros((max_features + 1, word_vec_dim))
for word, index in sequence_tokenizer.word_index.items():
    if False:
        pass
    # if word in word2vec.vocab:
    #         weights[index, :] = word2vec[word]
    else:
        not_ct += 1
        weights[index, :] = numpy.random.uniform(-0.25, 0.25, word_vec_dim)
# del word2vec
# del sequence_tokenizer
print(not_ct)

# ####  학습할 레이러를 구성한다.

# In[ ]:


embedded = keras.layers.Embedding(input_dim=max_features + 1,
                                  output_dim=word_vec_dim, input_length=input_dim,
                                  weights=[weights], trainable=True)(input_tensor)
# embedded2 = keras.layers.Dropout(0.9)(embedded)
# embedded2 = embedded


# In[ ]:


tensors = []
for filter_length in [3, 5]:
    tensor = keras.layers.Convolution1D(nb_filter=50, filter_length=filter_length)(embedded)
    tensor = keras.layers.Activation('relu')(tensor)
    tensor = keras.layers.MaxPooling1D(pool_length=input_dim - filter_length + 1)(tensor)
    tensor = keras.layers.Flatten()(tensor)
    tensors.append(tensor)

# In[ ]:


# embedded = keras.layers.Dropout(0.5)(embedded)
output_tensor = keras.layers.merge(tensors, mode='concat', concat_axis=1)
# output_tensor = keras.layers.Dropout(0.5)(output_tensor) # 0.7312
output_tensor = keras.layers.Dropout(0.5)(output_tensor)
output_tensor = keras.layers.Dense(len(set(y_list)), activation='softmax')(output_tensor)

# output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn) # See equations (6) and (7).

cnn = keras.models.Model(input_tensor, output_tensor)
cnn.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
print(cnn.summary())


# In[ ]:


cnn.fit(train, np.asarray(to_categorical(y_train)), batch_size=1000, nb_epoch=1,
        validation_data=(test, np.asarray(to_categorical(y_test))))

# In[ ]:


eval_x_text_list = []
with open("soma8_test_data.dat", encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        eval_x_text_list.append((info['pid'], " ".join(hannanum.nouns(info['name']))))

eval_x_list = texts_to_sequences2(map(lambda i: i[1], eval_x_text_list), sequence_tokenizer)

# pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],eval_x_text_list)))
pred = cnn.predict(eval_x_list)
pred_list = [argmax(y) for y in pred]

# In[ ]:


name = '염승우'
nickname = 'lokihardt'
mode = 'test'
param = {'pred_list': ",".join(map(lambda i: str(int(i)), pred_list)),
         'name': name, 'nickname': nickname, 'mode': mode}
d = requests.post('http://eval.buzzni.net:20001/eval', data=param)
print(d.json())


# ### CNN 으로 추출한 이미지 데이터 사용하기
#  * 이 부분은 각자 한번 해보도록 해요
