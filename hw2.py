import numpy as np
import argparse, os, re, functools
from stemmer import stemDoc

#Naive Bayes Classifier
def getArgs():
    p = argparse.ArgumentParser(description='Read and proceess a series of txt files in a directory to one clean stemmed text file.')
    p.add_argument('ham', help='An already stemmed text file containing all of the words in all the HAM emails in the TRAINING set')
    p.add_argument('spam', help='An already stemmed text file containing all of the words in all the SPAM emails in the TRAINING set')
    p.add_argument('hamDir', help='The directory where all the HAM emails in the TEST set are located')
    p.add_argument('spamDir', help='The directory where all the SPAM emails in the TEST set are located')
    p.add_argument('stopWords',nargs='?',default ='', help='file of stopwords to ignore when stemming')
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
    probList.append(prior)
    logList = map((lambda x: np.log(x)), probList)
    #print("loglist: {}".format(logList))
    prodRes = sum(logList)
    return prodRes

def genProbFromList(list, hamBag, spamBag, HorS):
    #from email, take each word, calculate prob, add to list
    percList = []
    totNumWordsInHam = sum(hamBag.values())
    totUniqueWordsInHam = len(hamBag)
    totNumWordsInSpam = sum(spamBag.values())
    totUniqueWordsInSpam = len(spamBag)
    for word in list:
        #calc probability from word
        hInst=hamBag.get(word, 0)
        sInst=spamBag.get(word, 0)
        #Laplace smoothing of 1
        if (hInst == 0 and HorS == "HAM") or (sInst == 0 and HorS == "SPAM"):
            #P(word|ham)laplace = 1 / totNumWordsinHam + totalUniqeWordInHam
            frac = 1/(totNumWordsInHam + totUniqueWordsInHam) if HorS == "HAM" else 1/(totNumWordsInSpam + totUniqueWordsInSpam) 
            percList.append(frac)
            #pr = "NOTFOUND {}/{},{}".format(hInst, totNumWordsInHam + totUniqueWordsInHam,frac) if HorS == "HAM" else "NOTFOUND {}/{}, {}".format(sInst, totNumWordsInSpam + totUniqueWordsInSpam,frac)
        else: 
            #P(word|ham)laplace = #ofWordInHam + 1 / totNumWordsinHam + totalUniqeWordInHam
            frac = (hInst+1)/(totNumWordsInHam + totUniqueWordsInHam) if HorS == "HAM" else (sInst+1)/(totNumWordsInSpam + totUniqueWordsInSpam) 
            percList.append(frac)
            #pr = "   FOUND {}/{}, {}".format(hInst, totNumWordsInHam + totUniqueWordsInHam, frac) if HorS == "HAM" else "   FOUND {}/{}, {}".format(sInst, totNumWordsInSpam + totUniqueWordsInSpam, frac)
    return percList

def test(dir,hamBag, spamBag, HorS, hprior, sprior,stopwords=""):
    correct = 0
    total = 0
    #for each email in dir
    for doc in os.listdir(dir):
        total += 1
        #print("stemming {}\{}".format(dir,doc))
        l = stemDoc(dir+doc) if stopwords=="" else stemDoc(dir+doc,stopwords)
        #print("CALC HAM PROB ON TEST EMAIL")
        l2 = genProbFromList(l, hamBag, spamBag, "HAM")
        #print("CALC SPAM PROB ON TEST EMAIL")
        l3 = genProbFromList(l, hamBag, spamBag, "SPAM")
        #print("hprior:{} list:{}\n".format(hprior, l2))
        #print("sprior:{} list:{}\n".format(sprior, l3))
        hres = calcCondProb(hprior, l2)
        sres = calcCondProb(sprior, l3)
        #print("hres:{} sres:{}".format(hres,sres))
        # if accurate add to counter, and total count
        if hres > sres and HorS == "HAM":
            correct+=1
            #print("HAM, {} > {}".format(hres,sres))
        elif hres < sres and HorS == "SPAM":
            correct+=1
            #print("SPAM, {} < {}".format(hres,sres))
    return correct/total
    #print("{}/{} correct, {:.2f}%".format(correct, total, (correct/total*100)))

#main function
if __name__ == "__main__":
    args = getArgs()
    ham = args.ham
    spam = args.spam
    hamDir = args.hamDir
    spamDir = args.spamDir
    stopWords = args.stopWords

    hamBag = initBag(ham)
    spamBag = initBag(spam)
    hamCount = len(os.listdir(hamDir))    
    spamCount = len(os.listdir(spamDir))    
    prior_ham = hamCount / (hamCount+spamCount)
    prior_spam = spamCount / (hamCount+spamCount)
    
    #Naive Bayes
    print("Running Naive Bayes with Laplace Smoothing of 1")
    ham_res = test(hamDir,hamBag,spamBag,"HAM",prior_ham,prior_spam,stopWords)
    spam_res = test(spamDir,hamBag,spamBag,"SPAM",prior_ham,prior_spam,stopWords)
    print("\tDetected {:.2f}% ham correctly".format(ham_res))
    print("\tDetected {:.2f}% spam correctly".format(spam_res))
