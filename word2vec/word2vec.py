import os
import codecs
import jieba
import jieba.analyse
import numpy as np
from gensim.models import word2vec

stopwords = {}.fromkeys(['的', '和', '是','\n', '-', '1'])
negPath = '/Python/SentimentAnalysis/neg.txt'
posPath = '/Python/SentimentAnalysis/pos.txt'
respath = '/Python/SentimentAnalysis/result.txt'

def cut_stopwords(line):
    res = ''
    for char in line:
        if char not in stopwords:
            res += char
    return res


def read_file_cut():
    comment = ''
    fileName = negPath
    source = open(fileName, encoding = 'utf-8')
    line = source.read()
    line = cut_stopwords(line)
    seglist = jieba.cut(line)
    output = ' '.join(seglist)
    comment += output
    fileName = posPath
    source = open(fileName, encoding = 'utf-8')
    line = source.read()
    line = cut_stopwords(line)
    seglist = jieba.cut(line)
    output = ' '.join(seglist)
    comment += output
    source.close()
    
    resName = respath
    if os.path.exists(resName):
            os.remove(resName)
    result = codecs.open(resName, 'w', encoding = 'utf-8')
    result.write(comment)
    result.close()

def model_train(model_name):
    sentences = word2vec.Text8Corpus(respath)
    model=word2vec.Word2Vec(sentences,size=200)
    model.save(model_name)
	# model.wv.save_word2vec_format(model_name + ".bin", binary=True)

def model_to_list(model):
    wordvec = model.wv
    wordlist = wordvec.index2word
    wvs = []
    wlen = len(wordlist)
    wvs = np.zeros((wlen, 200), dtype = 'float32')
    for i in range(len(wordlist)):
        a = wordlist[i]
        wvs[i] = model.wv[a]
    return wordlist, wvs

if __name__ == '__main__':
#     read_file_cut()
    model_name = 'word2vec.model'
    if not os.path.exists(model_name):
        model_train(model_name)
    model = word2vec.Word2Vec.load(model_name)
    wordlist, wvs = model_to_list(model)
    for i in range(10):
        print(wordlist[i], wvs[i])
