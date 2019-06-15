# Anomaly Detection Using Isolation Forest

## Project Description
The goal of this project was to implement the isolation forest algorithm for anomaly detection, which capitalizes on the notion that anomalies are few and different. Because of this, anomalies can be detected using an ensemble of trees by recursively splitting the data on a random feature and split point. Since anomalies have one or more features that make them unique, they are more susceptible to isolation and will have a shorter path length, on average.

This repository contains my python implementation of isolation forest optimized for both speed and sensitivity. Class definitions are provided for Node, IsolationTree, and IsolationTreeEnsemble, while methods are defined for fitting the Isotree and Isoforest, computing average path length, and calculating an anomaly score for each data point. Evaluation of the algorithm was performed using a script provided by Professor Terence Parr of USF, which tests for true positive rate (TPR), false positive rate (FPR), and fit time across three labeled datasets.
