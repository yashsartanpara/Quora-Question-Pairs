# Quora-Question-Pairs
Can you identify question pairs that have the same intent?

Main problem is to identify the questions which are same in given question pair.
Train and test data can be found at - <a href="https://www.kaggle.com/c/quora-question-pairs/data">Kaggle</a>

I have used word2vec and Glove model to create a word dictionary and used kares and siamese network to train a model.
Current accuracy of this model is ~67%

You can use spacy's pretrained model for word dictionary. Which is trained using wikipedia data. It can increase the accuracy.

Note: create a 'data' folder in cloned directory.
