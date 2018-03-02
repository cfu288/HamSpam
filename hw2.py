import numpy as np
import pandas as pd
import argparse, os, re, functools
from stemmer import stemDoc

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
    p.add_argument('hamDir', 
            help='The directory where all the HAM emails in the TEST set \
                    are located')
    p.add_argument('spamDir', 
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

def testNB(dir,hamBag, spamBag, HorS, hprior, sprior,stopwords=""):
    ''' (string, dict, dict, string, float, float, string) -> float

        Run a test on all of the emails in a given directory (dir),
        using bag of words for both ham (hamBag) and spam (spamBag).
        Return a percent of correctly classified emails given what 
        class they should be (HorS) and the ham prior (hprior) and
        spam prior (sprior).

    '''
    correct = 0
    total = 0
    #for each email in dir
    for doc in os.listdir(dir):
        total += 1
        l = stemDoc(dir+doc) if stopwords=="" else stemDoc(dir+doc,stopwords)
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

        Given the name of a stemmed text file (ham) generated from 
        all of the ham files, convert the document into a bag of 
        words stored as a dictionary and return the bag
        Bag of words here is slightly different than the one above
        as probability of word is stored instead of freq. Don't need 
        to do this, but saves time on calculations later
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
    return 1/(1+np.exp(-x))

def genDataArr(testHamDir, testSpamDir, uniqueWords, stopwords=""):
    numOfDocs = len(os.listdir(testHamDir)+os.listdir(testSpamDir)) #indexed rows
    numOfAttr = len(uniqueWords)+2 #column names
    uniqueWords.append("THRESHOLD")
    uniqueWords.append("CLASS")
    #print(uniqueWords)

    zero_data = np.zeros(shape=(numOfDocs,numOfAttr))
    df = pd.DataFrame(zero_data, columns=uniqueWords)
    #print(df)
    #print(df.loc[:,'spend'])
    
    ind = 0
    # For document in Ham dir
    for doc in os.listdir(testHamDir):
        if ind%10 == 0:
            print("processing doc {} of {}".format(ind,numOfDocs))
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
        df.loc[ind, "CLASS"] = 1
        df.loc[ind, "THRESHOLD"] = 1
        # Start at next document
        ind+=1

    #for doc in os.listdir(testSpamDir):
    #    print(doc)
        # Stem the document
    #    l = stemDoc(testSpamDir+doc) if stopwords==" \
    #            " else stemDoc(testSpamDir+doc,stopwords)
    #    bag = initBag1(l)
        # Move probablilites into df from bag
    #    for key in bag:
    #        try:
    #            df[ind, key] = bag[key]
    #        except:
    #            print("{} not found".format(key))
        # Set class to HAM and threshold to 1
    #    df[ind, "CLASS"] = 1
    #    df[ind, "THRESHOLD"] = 1
        # Start at next document
    #    ind+=1

    return df



#--------------MAIN--------------------

if __name__ == "__main__":
    # Get arguments
    args = getArgs()
    ham = args.ham
    spam = args.spam
    hamDir = args.hamDir
    spamDir = args.spamDir
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
    hamCount = len(os.listdir(hamDir))    
    spamCount = len(os.listdir(spamDir))    
    prior_ham = hamCount / (hamCount+spamCount) # number of hamDocs/totalDocs
    prior_spam = spamCount / (hamCount+spamCount)
    
    # Run Naive Bayes
    print("Running Naive Bayes with Laplace Smoothing of 1")
    #ham_res = testNB(hamDir,hamBag,spamBag,"HAM",prior_ham,prior_spam,stopWords)
    #spam_res = testNB(spamDir,hamBag,spamBag,"SPAM",prior_ham,prior_spam,stopWords)
    #print("\tDetected {:.2f}% ham correctly".format(ham_res))
    #print("\tDetected {:.2f}% spam correctly".format(spam_res))
    
    print("Running MCAP with L2")
    #l = stemDoc("minitest/ham/t1.txt")
    data_df = genDataArr('test/ham/','test/spam/',attrlst)
    print(data_df)
