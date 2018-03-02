import numpy as np
import pandas as pd
import argparse, os, re, functools
from stemmer import stemDoc
import random

#Naive Bayes Classifier
def getArgs():
    p = argparse.ArgumentParser(
            description='Read and proceess a series of txt files \
            in a directory to one clean stemmed text file.')
    p.add_argument('ham', 
            help='An already stemmed text file containing all of the \
            words in all the HAM emails in the TRAINING set')
    p.add_argument('spam', 
            help='An already stemmed text file containing all of the words \
            in all the SPAM emails in the TRAINING set')
    p.add_argument('testHamDir', 
            help='The directory where all the HAM emails in the TEST set \
                    are located')
    p.add_argument('testSpamDir', 
            help='The directory where all the SPAM emails in the TEST set \
            are located')
    p.add_argument('trainHamDir', 
            help='The directory where all the HAM emails in the TEST set \
                    are located')
    p.add_argument('trainSpamDir', 
            help='The directory where all the SPAM emails in the TEST set \
            are located')
    p.add_argument('stopWords',nargs='?',default ='', 
            help='(optional) file of stopwords to ignore when stemming')
    return p.parse_args()

#-----------------NAIVE BAYES-----------------

def initBag(ham):
    ''' (string) -> dict

        Given the name of a stemmed text file (ham) generated from 
        all of the ham files, convert the document into a bag of 
        words stored as a dictionary and return the bag
    '''
    d = {}
    with open(ham,'r') as openFile:
        for line in openFile:
            for word in line.split():
                if not word in d:
                    d[word] = 1
                else:
                    d[word] += 1
    return d

def calcCondProb(prior, probList):
    ''' (float, list) -> float

        Calculate the conditional probability of email being in a 
        class given the prior (prior) and a list of the probabilites 
        (probList) for each word in the email. Returns the log 
        probability of that email being part of a class.
    '''
    probList.append(prior)
    logList = map((lambda x: np.log(x)), probList)
    prodRes = sum(logList)
    return prodRes

def genProbFromList(listIn, hamBag, spamBag, HorS):
    ''' (list , dict, dict, string) -> list

        Given a list of all the words in one TEST email file 
        (listIn), return a new list of the probabilities of each 
        word appearing in either a HAM or SPAM email, depending on the 
        paramater (HorS). Laplace smoothing of 1 is applied to all 
        probabilities to avoid a zero probability for new words.
    '''
    #from email, take each word, calculate prob, add to list
    percList = []
    totNumWordsInHam = sum(hamBag.values())
    totUniqueWordsInHam = len(hamBag)
    totNumWordsInSpam = sum(spamBag.values())
    totUniqueWordsInSpam = len(spamBag)
    for word in listIn:
        #calc probability from word
        hInst=hamBag.get(word, 0)
        sInst=spamBag.get(word, 0)
        #Laplace smoothing of 1
        if (hInst == 0 and HorS == "HAM") or (sInst == 0 and HorS == "SPAM"):
            #P(word|ham)laplace = 1 / totNumWordsinHam + totalUniqeWordInHam
            frac = 1/(totNumWordsInHam + totUniqueWordsInHam
                    ) if HorS == "HAM" else 1/(
                            totNumWordsInSpam + totUniqueWordsInSpam) 
            percList.append(frac)
        else: 
            #P(word|ham)lap = #WdInHam+1/totNumWdsinHam+totalUniqeWdsInHam
            frac = (hInst+1)/(totNumWordsInHam + totUniqueWordsInHam
                    ) if HorS == "HAM" else (sInst+1)/(
                            totNumWordsInSpam + totUniqueWordsInSpam) 
            percList.append(frac)
    return percList

def testNB(testdir,hamBag, spamBag, HorS, hprior, sprior,stopwords=""):
    ''' (string, dict, dict, string, float, float, string) -> float

        Run a test on all of the emails in a given directory (testdir),
        using bag of words for both ham (hamBag) and spam (spamBag).
        Return a percent of correctly classified emails given what 
        class they should be (HorS) and the ham prior (hprior) and
        spam prior (sprior).

    '''
    correct = 0
    total = 0
    #for each email in dir
    for doc in os.listdir(testdir):
        total += 1
        l = stemDoc(testdir+doc) if stopwords=="" else stemDoc(testdir+doc,stopwords)
        l2 = genProbFromList(l, hamBag, spamBag, "HAM")
        l3 = genProbFromList(l, hamBag, spamBag, "SPAM")
        hres = calcCondProb(hprior, l2)
        sres = calcCondProb(sprior, l3)
        # if accurate add to counter, and total count
        if hres > sres and HorS == "HAM":
            correct+=1
        elif hres < sres and HorS == "SPAM":
            correct+=1
    return correct/total
    #print("{}/{} correct, {:.2f}%".format(correct, total, (correct/total*100)))

#--------------MCAP L2-------------------------------

def initBag1(doclist):
    ''' (list) -> dict

        Given the name of a stemmed list (doclist) generated from 
        a single email, convert the document into a bag of 
        words stored as a dictionary and return the bag.
        Bag of words here is slightly different than the one above
        as probability of word is stored instead of freq.
    '''
    d = {}
    for word in doclist:
        if not word in d:
            d[word] = 1
        else:
            d[word] += 1
    tot = sum(d.values())
    for key in d.keys():
       d[key] = d[key]/tot
    return d

def sigmoid(z):
    return 1/(1+np.exp(-z))

def genDataArr(testHamDir, testSpamDir, uniqueWords, stopwords=""):
    uniqueWords.append("THRESHOLD")
    uniqueWords.append("CLASS")
    numOfDocs = len(os.listdir(testHamDir)+os.listdir(testSpamDir)) #indexed rows
    numOfAttr = len(uniqueWords) #column names
    #print(uniqueWords)

    zero_data = np.zeros(shape=(numOfDocs,numOfAttr))
    df = pd.DataFrame(zero_data, columns=uniqueWords)
    #print(df)
    #print(df.loc[:,'spend'])
    
    ind = 0
    # For document in Ham dir
    for doc in os.listdir(testHamDir):
        if ind%10 == 0:
            print("processing doc {} of {}".format(ind,numOfDocs), end="\r")
        # Stem the document
        listFromDoc = stemDoc(testHamDir+doc) if stopwords=="\
                " else stemDoc(testHamDir+doc,stopwords)
        bag = initBag1(listFromDoc)
        #print("{} size = {}, bag size = {}".format(doc, len(listFromDoc), len(bag)))
        #print(listFromDoc)
        #print(bag)
        # Move probablilites into df from bag
        for key in bag:
            #print(df[ind,key])
            try:
                currentInd = df.loc[ind,key]
                #print("  "+str(df.loc[ind,key])+" set to " +str(bag[key]))
                df.loc[ind, key] = bag[key]
            except KeyError:
                #print("could not insert " + str(key))
                pass
        # Set class to HAM and threshold to 1
        df.loc[ind, "THRESHOLD"] = 1
        df.loc[ind, "CLASS"] = 1
        # Start at next document
        ind+=1

    for doc in os.listdir(testSpamDir):
        if ind%10 == 0:
            print("processing doc {} of {}".format(ind,numOfDocs),end="\r")
        # Stem the document
        listFromDoc = stemDoc(testSpamDir+doc) if stopwords=="\
                " else stemDoc(testSpamDir+doc,stopwords)
        bag = initBag1(listFromDoc)
        # Move probablilites into df from bag
        for key in bag:
            try:
                currentInd = df.loc[ind,key]
                df.loc[ind, key] = bag[key]
            except KeyError:
                pass
        # Set class to SPAM and threshold to 1
        df.loc[ind, "THRESHOLD"] = 1
        df.loc[ind, "CLASS"] = 0
        # Start at next document
        ind+=1

    return df

def mcap(df, itr, n, lamb):
    # Set size of pr to len of row count - num of docs
    Pr = np.random.rand(df.shape[0])
    # Set size of w to num of attr, - 2 for class and threshold
    w = np.random.rand(df.shape[1]-1)
    print("numofrow(col):{}".format(df.shape[0]))
    print("numofattr(col):{}".format(df.shape[1]))
    df2 = df[df.columns.difference(["CLASS"])]
    #print("pr(docs):{}, w(attr):{}".format(len(Pr), len(w)))
    for iteration in range(0,itr):
        print("\titeration {} : 0".format(iteration),end="\r")
        for x in range(0,df.shape[0]-1): # include all docs 
            print("\titeration {} : {}".format(iteration,x),end="\r")
            #print(df2)
            WxAttr = df2.loc[x,:].dot(w)
            Pr[x] = sigmoid(WxAttr)
            #break
            dw = np.zeros(df.shape[1]-1) # set size q to num of attr -1 for class
        i = 0
        for attr in list(df.columns.values):
            print("\titeration {} : {}".format(iteration,i),end="\r")
            if attr != "CLASS":
                for j in range(0, df.shape[0]-1):
                    #print("i:{}, j:{}".format(i,j))
                    classVal = df.loc[j,"CLASS"]
                    dw[i]=dw[i]+df.loc[j,attr]*(classVal- Pr[j])
            i+=1
        for i in range(0,df.shape[1]-2):
            w[i] = w[i]+n*(dw[i]-(lamb*w[i])) # Shift weights with regularization
    #print(list(df.columns.values))
    #print("w:{}".format(w))
    return w     

def testLR(testHamDir,testSpamDir,w, stopwords=""):
    total = 0
    totalCorrect = 0
    for doc in os.listdir(testHamDir):
        #print("TESTING " + doc)
        total += 1
        # Stem the document
        listFromDoc = stemDoc(testHamDir+doc) if stopwords=="\
                " else stemDoc(testHamDir+doc,stopwords)
        bag = initBag1(listFromDoc)
        resList = []
        # for word in bag, find resulting weight in dict, multiply the two, store in list
        resList.append(w["THRESHOLD"]) #append w0
        for key in bag.keys():
            #print("{}:w{}, b:{}".format(key,w[key],bag[key]))
            if key in w:
                resList.append(w[key]*bag[key])
        if sum(resList) > 1:
            totalCorrect += 1
    
    for doc in os.listdir(testSpamDir):
        total += 1
        # Stem the document
        listFromDoc = stemDoc(testSpamDir+doc) if stopwords=="\
               " else stemDoc(testSpamDir+doc,stopwords)
        bag = initBag1(listFromDoc)
        resList = []
       # for word in bag, find resulting weight in dict, multiply the two, store in list
        resList.append(w["THRESHOLD"]) #append w0
        for key in bag.keys():
            if key in w:
                resList.append(w[key]*bag[key])
        if sum(resList) < 1:
            totalCorrect += 1
    print("result is {}%".format(totalCorrect/total*100))
    return totalCorrect/total

#--------------MAIN--------------------

if __name__ == "__main__":
    # Get arguments
    args = getArgs()
    ham = args.ham
    spam = args.spam
    testHamDir = args.testHamDir
    testSpamDir = args.testSpamDir
    trainHamDir = args.trainHamDir
    trainSpamDir = args.trainSpamDir
    stopWords = args.stopWords

    # Initialize bags from stemmed test email files
    hamBag = initBag(ham)
    spamBag = initBag(spam)

    #generate list total uniqe words
    l = set(hamBag.keys())
    l2 = set(spamBag.keys())
    #total = len(hamBag) + len(spamBag) - len(l.intersection(l2))
    tot = l.union(l2)
    attrlst = list(tot)
    #print(total)
    # Calculate priors for NB
    hamCount = len(os.listdir(trainHamDir))    
    spamCount = len(os.listdir(trainSpamDir))    
    prior_ham = hamCount / (hamCount+spamCount) # number of hamDocs/totalDocs
    prior_spam = spamCount / (hamCount+spamCount)
    
    # Run Naive Bayes
    print("Running Naive Bayes with Laplace Smoothing of 1")
    #ham_res = testNB(testHamDir,hamBag,spamBag,"HAM",prior_ham,prior_spam,stopWords)
    #spam_res = testNB(testSpamDir,hamBag,spamBag,"SPAM",prior_ham,prior_spam,stopWords)
    #print("\tDetected {:.2f}% ham correctly".format(ham_res))
    #print("\tDetected {:.2f}% spam correctly".format(spam_res))
    
    print("Running MCAP with L2")
    # Get df, columns are attr, rows are docs
    data_df = None
    try:
        data_df = pd.read_csv("df.csv",index_col=0)
    except:
        data_df = genDataArr(trainHamDir,trainSpamDir,attrlst)
        print("\tsaving data_df to csv df.csv to save time on future runs")
        data_df.to_csv("df.csv")
    
    runForItr = 1 
    print("\tRunning MCAP with {} iterations".format(runForItr))
    w = mcap(data_df, runForItr, .02, 1)
    mapW = dict(zip(data_df.columns.values,w))
    print("\tTesting MCAP")
    testLR(testHamDir,testSpamDir,mapW)
    #testLR("minitest/tmph/","minitest/tmps/",mapW)
