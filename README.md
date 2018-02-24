## To run:
    python2 hw2.py <stemmedHam> <stemmedSpam> <testDirForHam> <testDirForSpam>
    * stemmedHam : An already stemmed text file containing all of the words in all the HAM emails in the TRAINING set.
    * stemmedSpam : An already stemmed text file containing all of the words in all the SPAM emails in the TRAINING set.
    * testDirForHam : The directory where all the HAM emails in the TEST set are located.
    * testDirForSpam : The directory where all the SPAM emails in the TEST set are located.
    * stopwords : (OPTIONAL) text file with stopwords to exclude from analysis.

## Example:
    ➜ python3 hw2.py stemmedFiles/train-ham-stemmed.txt stemmedFiles/train-spam-stemmed.txt  test/ham/ test/spam/
    Running Naive Bayes with Laplace Smoothing of 1
            Detected 0.95% ham correctly
            Detected 0.97% spam correctly

    ➜ python3 hw2.py stemmedFiles/train-ham-stemmed.txt stemmedFiles/train-spam-stemmed.txt test/ham/ test/spam/ stopwords.txt
    Running Naive Bayes with Laplace Smoothing of 1
            Detected 0.95% ham correctly
            Detected 0.97% spam correctly


