import numpy as np
import argparse, os, re, functools
from stemmer import stemDoc

#Naive Bayes Classifier

def getArgs():
    p = argparse.ArgumentParser(description='Read and proceess a series of txt files in a directory to one clean stemmed text file.')
    p.add_argument('ham', help='spam txt file')
    p.add_argument('spam', help='ham txt file')
    p.add_argument('hamDir', help='needed to count number of ham files')
    p.add_argument('spamDir', help='needed to count number of spam files')
    return p.parse_args()

def initBag(ham):
    d = {}
    with open(ham,'r') as openFile:
        for line in openFile:
            for word in line.split():
                if not word in d:
                    d[word] = 1
                else:
                    d[word] += 1
    return d

def chooseClass():
    pass

def calcCondProb(prior, probList):
    logList = map((lambda x: np.log(x)), probList)
    prodRes = functools.reduce((lambda x,y: x + y), logList)
    return prodRes*prior

def genProbFromEmail(emailFile, hamBag, spamBag, HorS="HAM"):
    #from email, take each word, calculate prob, add to list
    percList = []
    with open(emailFile, 'r') as openFile:
        for line in openFile:
            for word in line.split():
                #calc probability from word
                print(word)
                hInst=hamBag.get(word, 0)
                sInst=spamBag.get(word, 0)
                if hInst == 0 or sInst == 0:
                    percList.append(0)
                    print('0')
                else: 
                    frac = (hInst/(hInst+sInst)) if HorS == "HAM" else (sInst/(hInst+sInst))
                    percList.append(frac)
                    print(frac)
    return percList

#main function
if __name__ == "__main__":
    args = getArgs()
    ham = args.ham
    spam = args.spam
    hamDir = args.hamDir
    spamDir = args.spamDir

    hamBag = initBag(ham)
    spamBag = initBag(spam)
    hamCount = len(os.listdir(hamDir))    
    spamCount = len(os.listdir(spamDir))    
    prior_ham = hamCount / (hamCount+spamCount)
    prior_spam = spamCount / (hamCount+spamCount)
    
    l = genProbFromEmail("minitest_ham_stemmed.txt",hamBag,spamBag,"HAM");
    l2 = genProbFromEmail("minitest_ham_stemmed.txt",hamBag,spamBag,"SPAM");
    #print(calcCondProb(1,[.5,.5,.5]))
    #print(np.exp(calcCondProb(1,[.5,.5,.5])))
