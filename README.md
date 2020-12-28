# ActiveLink-for-Deep-Neural-Network

This repo contains the implementation of **ActiveLink** (*Ref: Ostapuk, N., et al. (2019). ActiveLink: Deep Active Learning for Link Prediction in Knowledge Graphs. The World Wide Web Conference on - WWW '19**:** 1398-1408.*). 

## Arguments 

```
--al-epochs
    number of iterations of active learning: (dataset_size * fraction_used) / sample_size
--batch-size
    training batch size
--dataset
    name of dataset
--embedding-dim
    number of embedding dimensions for entities and relations 
--early-stop-threshold
    stop training when trigger value is above this threshold (see below)
--eval-rate
    monitor model performance each N epochs (see below)
--inner-lr
    learning rate for inner update in meta-incremental training
--lr
    learning rate (meta-incremental training: learning rate for meta update)
--lr-decay
    learning rate decay
--model
    link prediction model, two options possible: ConvE or MLP
--n-clusters
    number of clusters for Structured Uncertainty sampling
--sample-size
    number of training examples per one AL iteration
--sampling-mode
    random, uncertainty, structured or structured-uncertainty
--training-mode
    retrain, incremental or meta-incremental
--window-size
    size of the window for meta-incremental training
```



## Repo Components 

+ Base code is adapted from https://github.com/eXascaleInfolab/ActiveLink (original repo)
  + but the implementation of **meta-incremental training** is re-implemented in neater code
  + some bugs are fixed as well
+ TransE training code, from https://github.com/thunlp/KB2E/blob/master/TransE/Train_TransE.cpp
+ Some zipped sample data 
  + Zip files include 
    + embedding, which is trained as mentioned in the original repo 
    + Entity2id and relation2id txt files 
    + train, test and valid files 
  + Dataset: FB15k-237, wikidata-300k



---

This repo is actively maintained. If any questions, feel free to contact via christang_1023@outlook.com or do a PR!   :P

