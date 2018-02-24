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
    p.add_argument('stopWords',nargs='?',default ='', help=' (optional) file of stopwords to ignore when stemming')
    return p.parse_args()

# Given a stemmed text file generated from all of the ham files, 
# convert the document into a bag of words (stored as a dictionary) 
# and return the bag
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

# Calculate the conditional probability of email being in a class
# given the prior and a list of the probabilites for each word in 
# the email. Returns the probability of email in class.
def calcCondProb(prior, probList):
    probList.append(prior)
    logList = map((lambda x: np.log(x)), probList)
    prodRes = sum(logList)
    return prodRes

# Given a list of all the words in one TEST email file,
# return a list of the probabilities of each word appearing
# in either a HAM or SPAM email. Laplace smoothing of 1 applied
# to all probabilities to avoid a zero probability for new words.
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
        else: 
            #P(word|ham)laplace = #ofWordInHam + 1 / totNumWordsinHam + totalUniqeWordInHam
            frac = (hInst+1)/(totNumWordsInHam + totUniqueWordsInHam) if HorS == "HAM" else (sInst+1)/(totNumWordsInSpam + totUniqueWordsInSpam) 
            percList.append(frac)
    return percList

# Run a test on all of the emails in a given directory,
# return a percent of correctly classified emails.
def test(dir,hamBag, spamBag, HorS, hprior, sprior,stopwords=""):
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

#main function
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

    # Calculate priors for NB
    hamCount = len(os.listdir(hamDir))    
    spamCount = len(os.listdir(spamDir))    
    prior_ham = hamCount / (hamCount+spamCount)
    prior_spam = spamCount / (hamCount+spamCount)
    
    # Run Naive Bayes
    print("Running Naive Bayes with Laplace Smoothing of 1")
    ham_res = test(hamDir,hamBag,spamBag,"HAM",prior_ham,prior_spam,stopWords)
    spam_res = test(spamDir,hamBag,spamBag,"SPAM",prior_ham,prior_spam,stopWords)
    print("\tDetected {:.2f}% ham correctly".format(ham_res))
    print("\tDetected {:.2f}% spam correctly".format(spam_res))
