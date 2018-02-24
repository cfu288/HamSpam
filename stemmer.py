import numpy as np
import argparse, os, re
from nltk.stem.porter import * 

def getArgs():
    p = argparse.ArgumentParser(description='Read and proceess a series of txt files in a directory to one clean stemmed text file.')
    p.add_argument('stemDir', help='directory of files to stem')
    p.add_argument('outFile', help='dir to output txt of stemmed files')
    p.add_argument('stopWords',nargs='?',default ='', help='file of stopwords to ignore when stemming')
    return p.parse_args()

def stemDoc(docDir,stopWords=""):
    stopWordsL = []
    if stopWords != "":
        with open(stopWords,'r', encoding='utf-8', errors='ignore') as stopFile:
            for line in stopFile:
                for word in line.split():
                    stopWordsL.append(word.lower())
     
    unstemmed_words_list = []
    regex = re.compile('[^a-zA-Z0-9]')
    with open(docDir,'r',encoding='utf-8', errors='ignore') as openFile:
        for line in openFile:
            for word in line.split():
                if word not in stopWords:
                    cleaned = regex.sub('', word)
                    if len(cleaned) > 0:
                        unstemmed_words_list.append(cleaned.lower())
    #print(unstemmed_words_list)
    stemmer = PorterStemmer()
    stemmed_words_list = [stemmer.stem(plural) for plural in unstemmed_words_list]
    return stemmed_words_list

#main function
if __name__ == "__main__":
    args = getArgs()
    stemDir  = args.stemDir
    outFile = args.outFile
    stopWords = args.stopWords

    regex = re.compile('[^a-zA-Z0-9]')
    
    stopWordsL = []
    if stopWords != "":
        with open(stopWords,'r', encoding='utf-8', errors='ignore') as stopFile:
            for line in stopFile:
                for word in line.split():
                    cleaned = regex.sub('', word)
                    stopWordsL.append(cleaned.lower())
        #with open("stopWordsCleaned.txt", 'w') as stemmed_file1:
        #    for word in stopWordsL:
        #        stemmed_file1.write(word + " ")
        
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
                        if word not in stopWordsL:
                            cleaned = regex.sub('', word)
                            if len(cleaned) > 0:
                                unstemmed_words_list.append(cleaned.lower())
    #print(unstemmed_words_list)
    stemmer = PorterStemmer()
    stemmed_words_list = [stemmer.stem(plural) for plural in unstemmed_words_list]
    #print(stemmed_words_list)

    if stopWords != "":
        l = outFile.split('.')
        with open(l[0]+"noStopWords"+".txt", 'w') as stemmed_file:
            for word in stemmed_words_list:
                stemmed_file.write(word + " ")
    else:
        with open(outFile, 'w') as stemmed_file:
            for word in stemmed_words_list:
                stemmed_file.write(word + " ")
