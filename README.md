# Kaggle-Quora-Incincere-question-classification
Kaggle -  Quora Insincere Questions Classification 

Repo to elaborate on two of my many attempts at solving [Quora Insincere Questions Classification competition](https://www.kaggle.com/c/quora-insincere-questions-classification)

### 1. An Attempt at using Siamese network: [LB-64.5]
#### Intuition :
My Intuition was to overcome the huge class imbalance in the dataset with the use of Siamese network. Instead of trying to understand sincere and insincere question, Siamese networks approaches the problem as a similarity problem. It tries to understand how similar a given pair is. Since it is easier to genrate as many combination of similar and non similar pairs as we would need from the dataset ( theoreticaly we can augment pairs in the order of Billions ). I thought the class imbalance would not be a problem and can lead to a better result.

#### Learning :
The 2 hour constrain of kaggle kernels restricted me from using only 3-5 million pairs of generated data. But these were unfortunately not sufficient for the network to understand the nuances of many ambigous and difficult questions. I learnt that siamese are a good choice when we have much lesser samples per class. As the sample size increases , the triplets / pair sizes increases quadratically making training impossible.

As the training size increases , the number of easy pairs ( pairs which are easier to differentiate / identify ) aslo increases in a given batch. This affects the gradient update as the average loss per batch is much less when majority of samples in a batch are easy pairs. This stalls the model from learning further after a certain point.

#### Things to try:
To use only hard samples during loss calculation. As stated in https://arxiv.org/pdf/1703.07737.pdf . For the dataset of this magnitude , I presume it will be still be impossible to get a decent result with a 2 hour time constraint. ( Implementation of Hard pair selector - https://github.com/adambielski/siamese-triplet/blob/master/utils.py )
#### References:
https://www.kaggle.com/shujian/single-rnn-with-4-folds-clr

### 2. Inverse ratio undersampled stacking [LB-69.1]:

My other attempt at handling the class imbalance is to use inverse ratio undersampled stacking, where in the dataset is split in to n parts with all the samples of minority class from the dataset being present in every part and the majority class is split in such a way that the majority class becomes the minority in each part. We train a model (in this case a bidirectional lstm-gru with skip connection and attention )on all these parts and make use of the prediction from these n models and build a meta classifier on top of these predictions.
