# Introduction:
## I. Project Name: Sentiment Analysis with Natural Language Processing.
## II. Technology Used: NLTK (Natural Language tool kit), Python.
## III. Supported Operating System: 
This project can easily be run in Windows operating system. To run this project 
Spider of Anaconda python need to be installed.
## IV. Project Goals: 
a. Compare time complexity of different classifier for sentiment analysis.
b. To improve accuracy of different classifier by using updated dataset.
c. Compare accuracy among those classifiers.
## V. Project Objectives:
Sentiment Analysis is one of the greatest applications of Computer Science. 
Sentiment Analysis is an application of Natural Language processing which is 
used to identify the opinion of a person about the product or discussing topic; 
whether it’s negative, positive or neutral. It has a great importance in social 
media and business sector. In social media, it is used to measure the statistic. In 
business sector, marketing strategy of company, improvements of campaign 
success of a product, improvement of product messaging are greatly depending 
on it. Our task to develop a web based application to determine the sensitivity 
of the user input text which is as comment. There are many well-known 
techniques to solve this problem. As a beginner, I have chosen Naive Bayes 
classifier. This technique is easy to understand and implement. The accuracy 
of this technique is not very good but enough to work with it.
## Sentiment Analysis: 
Sentiment Analysis is the process of determining whether a piece of 
writing is positive, negative or neutral. It’s also known as opinion mining, 
deriving the opinion or attitude of a speaker.
## Naïve Bayes Classifier: 
In this section, Naive Bayes classifier and its application to sentiment analysis, 
Text Parsing, Feature Vector are discussed. 
Naive Bayes Classifier is enough powerful algorithm for classification task. It 
is based on Bayes Theorem. Bayes Theorem is a well-known in the field of 
probability. Bayes Theorem works on condition probability. Conditional 
probability refers to the probability of happening of an event depends on the 
occurrence of another event which has already been.
If B1, B2, B3, ........., Bn are mutually exclusive events of sample space S 
where S = B1 ∪ B2 ∪ B3 ∪ ..... ∪ Bn
and P(Bi) > 0;
where i=1, 2, 3, ....... ,n. 
If A is another event of the sample space S such that P(A) > 0.
P(Bi |A)= P(Bi|A) =
𝑃 𝐵𝑖 𝑃(𝐴|𝐵𝑖)
 𝑃 𝐵𝑖 𝑃(𝐴|𝐵𝑖)
𝑛
𝑖=1
where i=1,2,3,.....n. This is the Bayes Theorem.
![image](https://github.com/ZobairHussain/Sentiment-Analysis/assets/49007316/ef94586a-9b58-4587-8fbc-fc0976e4fae2)

## Data Set:
Data set is taken from Amazon short movie review.
Dataset link: http://jmcauley.ucsd.edu/data/amazon/
Evaluation metrics:
As a classification problem, Sentiment Analysis uses the evaluation metrics of 
Precision, Recall, F-score, and Accuracy. Also, average measures like macro, micro, 
and weighted F1-scores are useful for multi-class problems. Depending on the balance 
of classes of the dataset the most appropriate metric should be used.
![image](https://github.com/ZobairHussain/Sentiment-Analysis/assets/49007316/7f542488-7c5a-47a0-87d8-f80343aad41c)

## K Fold Cross Validation:
Cross-validation is a technique to evaluate predictive models by partitioning 
the original sample into a training set to train the model, and a test set to 
evaluate it.
In k-fold cross-validation, the original sample is randomly partitioned into k 
equal size subsamples. Of the k subsamples, a single subsample is 
retained as the validation data for testing the model, and the remaining k-1 
subsamples are used as training data. The cross-validation process is then 
repeated k times (the folds), with each of the k subsamples used exactly 
once as the validation data. The k results from the folds can then be 
averaged (or otherwise combined) to produce a single estimation. The 
advantage of this method is that all observations are used for both training 
and validation, and each observation is used for validation exactly once.
For classification problems, one typically uses stratified k-fold crossvalidation, in which the folds are selected so that each fold contains roughly 
the same proportions of class labels.
In repeated cross-validation, the cross-validation procedure is repeated n 
times, yielding n random partitions of the original sample. The n results are 
again averaged (or otherwise combined) to produce a single estimation.

## Conclusion: From this project we can see that some classifier gives more accuracy 
and some less. But by seeing only accuracy we can’t say which classifier is best. 
Because there are more factors like space complexity, time complexity etc. at the 
end, it can be said that, in the point of view of all attribute Naïve Bayes Classifier 
is better.
