import os
import codecs
import jieba
import jieba.analyse

stopwords = {}.fromkeys([',', '.', '的', '和', '是',  '。', '，'])
path = '/Python/情感分析/2000/neg/neg.'
respath = '/Python/情感分析/result/neg.'
NUM = 999


def read_file_cut():
    num = 0
    while num <= NUM:
        fileName = path + str(num) + '.txt'
        resName = respath + str(num) + '.txt'
        source = open(fileName, encoding = 'utf-8')
        if os.path.exists(resName):
            os.remove(resName)
        result = codecs.open(resName, 'w', encoding = 'utf-8')
        line = source.read()
        final = ''
        for char in line:
            if char not in stopwords:
                final += char
        seglist = jieba.cut(final)
        output = ' '.join(seglist)
        keywords = jieba.analyse.extract_tags(output)
        output = ' '.join(keywords)
        result.write(output)
        num = num + 1
        
if __name__ == '__main__':
    read_file_cut()
