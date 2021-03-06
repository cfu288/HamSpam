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
    '''
    (string,string,list,string)-> pandas dataframe
    Generate a dataframe storing the probabilities of each word in each document.
    This DF is later used in the MCAP algorithm. Takes the directory of training dir
    for ham and spam (inaccuratly labeled testHamDir and testSpamDir here) as well as
    a list of unique attributes to build the df. Stopwords are optional
    '''
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
    '''(df, int, int, int) -> arr
    Takes in dataframe and 3 ints for the weight function, returns
    an array of weights to be used.
    '''
    # Set size of pr to len of row count - num of docs
    #Pr = np.random.rand(df.shape[0])
    Pr = np.full(df.shape[0],.5)
    # Set size of w to num of attr, - 2 for class and threshold
    #w = np.random.rand(df.shape[1]-1)
    w = np.full(df.shape[1]-1, .5)
    df2 = df[df.columns.difference(["CLASS"])]
    for iteration in range(0,itr):
        # Calculate Pr[i] = Pr(class=1|Data[i],w)
        for x in range(0,df.shape[0]): # include all docs 
            WxAttr = df2.loc[x,:].dot(w)
            #print("DOC {} DP: {}".format(x, WxAttr))
            Pr[x] = sigmoid(WxAttr)
            #print("DOC {} Sig: {}".format(x, Pr[x]))
        # Array dw[0..n] init to 0
        dw = np.zeros(df.shape[1]-1) # set size q to num of attr -1 for class
        
        i = 0 #col/attr selector
        for attr in list(df.columns.values):
            if attr != "CLASS":
                for j in range(0, df.shape[0]): # for each example doc of attr
                    classVal = df.loc[j,"CLASS"]
                    dw[i] = dw[i] + df.loc[j,attr] * (classVal - Pr[j])
                #print("attr {} has dw[{}] : {}".format(attr,i,dw[i]))
            i+=1
        for i in range(0,df.shape[1]-1):
            w[i] = w[i] + n*(dw[i]-(lamb*w[i])) # Shift weights with regularization
        #print("W on {} iteration : {}".format(iteration,w))
        #print("Pr on {} iteration : {}".format(iteration, Pr))
        #print("dw on {} iteration : {}".format(iteration, dw))
    return w     

def testLR(testHamDir,testSpamDir,w, stopwords=""):
    '''(string,string,dict,string) -> float
    takes two dir names in string form that hold the test emails, a dict with the 
    trained weights from MCAP, and an optional txt file with stop words. It returns
    the percent of test emails it correctly predicts.
    '''
    total = 0
    totalCorrect = 0
    for doc in os.listdir(testHamDir):
        total += 1
        # Stem the document
        listFromDoc = stemDoc(testHamDir+doc) if stopwords=="\
                " else stemDoc(testHamDir+doc,stopwords)
        bag = initBag1(listFromDoc)
        resList = []
        # for word in bag, find resulting weight in dict, multiply the two, store in list
        resList.append(w["THRESHOLD"]) #append w0
        print("H DOC {}:".format(doc))
        print("  word {}, val {} ".format("threshold",w["THRESHOLD"]))
        for key in bag.keys():
            if key in w:
                print("  word {}, freq in doc {}, weight {} ".format(key,bag[key],w[key]))
                resList.append(w[key]*bag[key])
        print("  {} has sig {}".format(doc,sigmoid(sum(resList))))
        if sigmoid(sum(resList)) > .5:
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
        print("S DOC {}:".format(doc))
        print("  word {}, val {} ".format("threshold",w["THRESHOLD"]))
        for key in bag.keys():
            if key in w:
                print("  word {}, freq in doc {}, weight {} ".format(key,bag[key],w[key]))
                resList.append(w[key]*bag[key])
        print("  {} has sig {}".format(doc,sigmoid(sum(resList))))
        if sigmoid(sum(resList)) < .5:
            totalCorrect += 1
    #print("{}/{}".format(totalCorrect,total))
    print("  \n\n")
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
    
    # Calculate priors for NB
    hamCount = len(os.listdir(trainHamDir))    
    spamCount = len(os.listdir(trainSpamDir))    
    prior_ham = hamCount / (hamCount+spamCount) # number of hamDocs/totalDocs
    prior_spam = spamCount / (hamCount+spamCount)
    
    # Run Naive Bayes
    print("Running Naive Bayes with Laplace Smoothing of 1")
    ham_res = testNB(testHamDir,hamBag,spamBag,"HAM",prior_ham,prior_spam,stopWords)
    spam_res = testNB(testSpamDir,hamBag,spamBag,"SPAM",prior_ham,prior_spam,stopWords)
    print("\tDetected {:.2f}% ham correctly".format(ham_res*100))
    print("\tDetected {:.2f}% spam correctly".format(spam_res*100))
    
    # Run Logistic Regression
    print("Running MCAP with L2")
    # Get df, columns are attr, rows are docs
    data_df = None
    try:
        data_df = pd.read_csv("df.csv",index_col=0)
    except:
        data_df = genDataArr(trainHamDir,trainSpamDir,attrlst,stopWords)
        print("\tSaving data_df to csv df.csv to save time on future runs")
        data_df.to_csv("df.csv")
    
    print(data_df)
    runForItr = 10
    print("\tRunning MCAP with {} iterations".format(runForItr))
    w = mcap(data_df, runForItr, .1, 1)
    mapW = dict(zip(data_df.columns.values,w))
    #print(mapW)
    print("\tTesting MCAP")
    res = testLR(testHamDir,testSpamDir,mapW,stopWords)
    print("Detected {:.2f}% correctly".format(res*100))
    #testLR("minitest/tmph/","minitest/tmps/",mapW)
