#encoding=utf-8
import nltk
from gensim.models import word2vec
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
'''
tf.nn.embedding_lookup(word_vecs,sen_index)
word_vecs是训练数据集中所有单词通过word2vec训练出的词向量（注意：第一行是全0，以便于对句子进行补0，最后一行也是全0，以便于为数组中不存在的单词赋值）
sen_index一共有batch_size行sentence_length列，每行数据是一个文本中的各个单词在word_vecs中所对应的索引行
tf.nn.embedding_lookup()所执行的操作是把sen_index的每一行转换为一列进行独热编码然后和word_vecs进行相乘，以此来根据索引取出相应行
'''
#获得英文停用词
def get_stop_words():
    with open('data/stoplist.csv','r') as st:
        word=st.read()
        return word.splitlines()
#从数据文件中获得数据与类别标签，并完成分词以及去除停用词的操作
def get_train_data():
    label = []
    word_cut_list = []
    stop_word = get_stop_words()
    sent_list = []
    with open('data/newdata.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            newline = ' '.join(line.strip().split())
            split_list = newline.split('senlabel')
            sent_list.append(split_list[0])
            label.append(split_list[1])
    word_list = [nltk.word_tokenize(sen) for sen in sent_list]
    label = list(map(int, label))
    for sen in word_list:
        word_cut_list.append([word for word in sen if word not in stop_word])
    return word_cut_list,label
def get_onehot_label():
    _, label=get_train_data()
    ont_hot = OneHotEncoder()
    label = ont_hot.fit_transform(np.array(label).reshape([-1, 1]))
    label = label.toarray()
    return label
#结合所有数据训练词向量
def train_model():
    word_cut_list, label=get_train_data()
    #注意word_cut_list是一个大List里面有小List,List中的数据是一个个字符串
    #hs用于控制选用何种算法，window用于控制训练词向量时和前后几个单词相关,size是每个单词用多少维表示
    word2vec_model = word2vec.Word2Vec(word_cut_list, hs=1, min_count=1, window=2, size=80)
    word2vec_model.save('word2vec_model/text_cnn_model')
#获得所有单词的词向量并完成单词向索引的映射
def word_pro(vec_size):
    word2vec_model=word2vec.Word2Vec.load('word2vec_model/text_cnn_model')
    index_dict={}
    word_vecs=[]
    word_cut_list,_=get_train_data()
    #注意下面的这一步，它是为了方便对那些不满足规定句子长度的句子进行补0操作
    word_vecs.append(list(np.zeros((vec_size,))))
    for sen in word_cut_list:
        for word in sen:
            index_dict[word]=len(word_vecs)
            word_vecs.append(list(word2vec_model[word]))
    index_dict['UNKNOW']=len(word_vecs)
    word_vecs.append(list(np.zeros((vec_size,))))
    #注意此处要转换为float32
    word_vecs=np.array(word_vecs).astype(np.float32)
    return word_vecs,index_dict
#把分词，去停用词后的数据转换为索引
def get_index_array(sen_length,vector_size):
    word_cut_list, _ = get_train_data()
    _, index_dict=word_pro(vector_size)
    sen_index=[]
    for sen in word_cut_list:
        word_index = []
        for word in sen:
            if word in list(index_dict.keys()):
                word_index.append(index_dict[word])
            else:
                word_index.append(index_dict['UNKNOW'])
        sen_len=len(sen)
        if sen_len<sen_length:
            word_index.extend([0]*(sen_length-sen_len))
        sen_index.append(word_index)
    #索引值，保证代码健壮性，向整型转换
    sen_index=np.array(sen_index).astype(np.int32)
    return sen_index