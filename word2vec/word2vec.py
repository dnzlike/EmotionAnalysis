import os
import codecs
import jieba
import jieba.analyse
import numpy as np
import multiprocessing
from gensim.models import word2vec

stopwords = {}.fromkeys(['\n', '\''])
negPath = './10000/neg/neg.'
posPath = './10000/pos/pos.'
respath = '/Python/SentimentAnalysis/result.txt'
NEG = 2999
POS = 6999

def cut_stopwords(line):
    res = ''
    for char in line:
        if char not in stopwords:
            res += char
    return res


def read_file_cut():
    neg = 0
    pos = 0
    text = []
    while neg <= NEG:
        fileName = negPath + str(neg) + '.txt'
        source = open(fileName, encoding = 'utf-8')
        line = source.read()
        line = cut_stopwords(line)
        text.append(line)
        source.close()
        neg = neg + 1
    while pos <= POS:
        fileName = posPath + str(pos) + '.txt'
        source = open(fileName, encoding = 'utf-8')
        line = source.read()
        line = cut_stopwords(line)
        text.append(line)
        source.close()
        pos = pos + 1
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    print(text[:10])
    print(len(text))
    return text

def model_train(model_name):
    sentences = word2vec.Text8Corpus(respath)
    model=word2vec.Word2Vec(sentences,size=100, min_count=10, window=7, iter=1)
    model.save(model_name)
	# model.wv.save_word2vec_format(model_name + ".bin", binary=True)

def model_to_list(model):
    wordvec = model.wv
    wordlist = wordvec.index2word
    wvs = []
    wlen = len(wordlist)
    wvs = np.zeros((wlen, 100), dtype = 'float32')
    for i in range(len(wordlist)):
        a = wordlist[i]
        wvs[i] = model.wv[a]
    return wordlist, wvs



if __name__ == '__main__':
    text = read_file_cut()
    
    '''model_name = 'word2vec.model'
    if not os.path.exists(model_name):
        model_train(model_name)
    model = word2vec.Word2Vec.load(model_name)
    wordlist, wvs = model_to_list(model)
    for i in range(10):
        print(wordlist[i], wvs[i])'''
