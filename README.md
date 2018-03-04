## To run:
    python3 hw2.py <stemmedHam> <stemmedSpam> <testDirForHam> <testDirForSpam> <trainDirForHam> <trainDirForSpam> <stopwords>
    * stemmedHam : An already stemmed text file containing all of the words in all the HAM emails in the TRAINING set.
    * stemmedSpam : An already stemmed text file containing all of the words in all the SPAM emails in the TRAINING set.
    * testDirForHam : The directory where all the HAM emails in the TEST set are located.
    * testDirForSpam : The directory where all the SPAM emails in the TEST set are located.
    * trainDirForHam : The directory where all the HAM emails in the TRAIN set are located.
    * trainDirForSpam : The directory where all the SPAM emails in the TRAIN set are located.
    * stopwords : (OPTIONAL) text file with stopwords to exclude from analysis.

## Example:
    // Learning rate .02, lambda 1
    ➜ python3 hw2.py stemmedFiles/train-ham-stemmed.txt stemmedFiles/train-spam-stemmed.txt test/ham/ test/spam/ train/ham/ train/spam/
    Running Naive Bayes with Laplace Smoothing of 1
        Detected 95.40% ham correctly
        Detected 96.92% spam correctly
    Running MCAP with L2
        Running MCAP with 1 iterations
        Testing MCAP
        Detected 72.80% correctly
    
    // With stemming
    ➜ python3 hw2.py stemmedFiles/train-ham-stemmed-no-stopwords.txt stemmedFiles/train-spam-stemmed-no-stopwords.txt test/ham/ test/spam/ train/ham/ train/spam/ stopwords.txt 
    Running Naive Bayes with Laplace Smoothing of 1
        Detected 88% ham correctly
        Detected 98% spam correctly
    Running MCAP with L2
        Running MCAP with 1 iterations
        Testing MCAP
        Detected 72.80% correctly
 
## Academic Honesty:
    I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense.

## Discussion
    My Naive Bayes algoritm ran as intended, and scored a 95% correct on ham emails and 96% correct on spam emails. Surprisingly enough, stopwords reduced my performance to 88% on ham documents and increased to 98% on spam documents. I implemented BagOfWords using dictionarys, with words being the keys and frequency in all the document being the value. I assume this drop comes down to the fact that the words I removed actually are relevant to determining ham emails, and that their removal degraded performance.I happened to choose to report the percentages sepeerately for this example for no real reason other than that is how I structured my NB functions. I change this in LR and report only the total value. 

    Note, the first time you run the script, it will take a while to generate the data strucure/matrix that will be usedin MCAP. After it has been created the first time, the data struct will be written to df.csv for quick runs in the future instead of reading all the files.

    I had several issues with my LR implementation. While I followed the MCAP algorithm layed out on the slides provided, I think I made a poor choice of data structures to use. I used a panda's dataframe as my data matrix, however it gave me very poor times. This might be due to my unfamiliarity with pandas, but even one iteration takes about 7 minutes on my local machine. For that reason, my performance was much worse on MCAP, as I limited my iterations to one runthrough of the documents (1 iteration includes running though all 400+ documents once and ~8000 attributes since there are around 8000 words). I just left the iterations hard coded because more than 1 seems difficult anyways. Because of my fewer iterations, it is likely that I never reach convergence. I've run my implementation on smaller datasets with higher iterations, where it has run decently. However I still have issues with the current dataset. In the future, I'll make sure to check be more careful which tools I use and how I use them. 

Now to run the tests:
    - Running with a learning rate of .02 and lambda .0001, I received a 72.80%  
    - Running with a learning rate of .02 and lambda .1, I received a 72.80% 
    - Running with a learning rate of .02 and lambda 1, I received a 72.80%
    - Running with a learning rate of .02 and lambda 3, I received a 72.80%  
    - Running with a learning rate of .02 and lambda 5, I received a 72.80%   
    - Running with a learning rate of .1 and lambda 2, I received a 72.80%
    - Running with a learning rate of 1 and lambda 5, I received a 27.20%
I'm not really sure how to interpret these results. None of my lambda changes seem to have an impact on the final results. I'm starting to question my implementations validity

Similarly, removing stopwords does not seem to change anything

## Code Explation
Comments in hw2.py should be relatively comprehensive and explain what I did.
