import os
import codecs
import jieba
import jieba.analyse
from gensim.models import word2vec

stopwords = {}.fromkeys(['的', '和', '是'])
path = '/Python/SentimentAnalysis/2000/neg/neg.'
respath = '/Python/SentimentAnalysis/result/neg.'
NUM = 999

def cut_stopwords(line):
    res = ''
    for char in line:
        if char not in stopwords:
            res += char
    return res


def read_file_cut():
    num = 0
    comment = ''
    while num <= NUM:
        fileName = path + str(num) + '.txt'
        source = open(fileName, encoding = 'utf-8')
        line = source.read()
        line = cut_stopwords(line)
        seglist = jieba.cut(line)
        output = ' '.join(seglist)
        comment += output
        source.close()
        num = num + 1
    resName = respath + 'all.txt'
    if os.path.exists(resName):
            os.remove(resName)
    result = codecs.open(resName, 'w', encoding = 'utf-8')
    result.write(comment)
    result.close()

def word_to_vec():
    sentences = word2vec.Text8Corpus(respath + 'all.txt')
    model=word2vec.Word2Vec(sentences,min_count=3, size=50, window=5, workers=4)
    for i in model.most_similar(u"房间"):
        print(i[0],i[1])
        
if __name__ == '__main__':
    read_file_cut()
    word_to_vec()
