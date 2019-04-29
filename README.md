# Anomaly Detection Using Isolation Forest

## Project Description
The goal of this project was to implement the isolation forest algorithm for anomaly detection, which capitalizes on the notion that anomalies are few and different. Because of this, anomalies can be detected 
using an ensemble of trees by recursively splitting the data on a random feature and random split point. Since anomalies have one or more features that make them unique, they are more susceptible to isolation and will have a shorter path length, on average.

This repository contains my python implementation of isolation forest, including class definitions for Node, IsolationTree, and IsolationTreeEnsemble. Methods are defined for fitting the isotree and isoforest, computing path length, as well as calculating an anomaly score for each point.
