import numpy as np
import argparse, os, re
from nltk.stem.porter import * 

def getArgs():
    p = argparse.ArgumentParser(description='Read and proceess a series of txt files in a directory to one clean stemmed text file.')
    p.add_argument('stemDir', help='directory of files to stem')
    p.add_argument('outFile', help='dir to output txt of stemmed files')
    return p.parse_args()

def stemDoc(docDir):
    unstemmed_words_list = []
    regex = re.compile('[^a-zA-Z0-9]')
    with open(docDir,'r',encoding='utf-8', errors='ignore') as openFile:
        for line in openFile:
            for word in line.split():
                cleaned = regex.sub('', word)
                if len(cleaned) > 0:
                    unstemmed_words_list.append(cleaned.lower())
    #print(unstemmed_words_list)
    stemmer = PorterStemmer()
    stemmed_words_list = [stemmer.stem(plural) for plural in unstemmed_words_list]
    return stemmed_words_list

def stemDir(stemDir):
    unstemmed_words_list = []
    regex = re.compile('[^a-zA-Z0-9]')
    for filename in os.listdir(stemDir):
        if filename.endswith(".txt"):
            filedir = stemDir+'/'+filename
            with open(filedir,'r',encoding='utf-8', errors='ignore') as openFile:
                for line in openFile:
                    for word in line.split():
                        cleaned = regex.sub('', word)
                        if len(cleaned) > 0:
                            unstemmed_words_list.append(cleaned.lower())
    stemmer = PorterStemmer()
    stemmed_words_list = [stemmer.stem(plural) for plural in unstemmed_words_list]
    return stemmed_words_list


#main function
if __name__ == "__main__":
    args = getArgs()
    stemDir  = args.stemDir
    outFile = args.outFile

    regex = re.compile('[^a-zA-Z0-9]')
    #list of words in all txt documents in folder
    unstemmed_words_list = []
    #for current file in dir
    for filename in os.listdir(stemDir):
        if filename.endswith(".txt"):
            filedir = stemDir+'/'+filename
            #print(filedir)
            with open(filedir,'r',encoding='utf-8', errors='ignore') as openFile:
                for line in openFile:
                    for word in line.split():
                        cleaned = regex.sub('', word)
                        if len(cleaned) > 0:
                            unstemmed_words_list.append(cleaned.lower())
    #print(unstemmed_words_list)
    stemmer = PorterStemmer()
    stemmed_words_list = [stemmer.stem(plural) for plural in unstemmed_words_list]
    #print(stemmed_words_list)

    with open(outFile, 'w') as stemmed_file:
        for word in stemmed_words_list:
            stemmed_file.write(word + " ")
