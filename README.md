# A Random Walk Through The Forest
## Predicting Online Recipe Ranking Using Decision Trees and Random Forests

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk/master/dail.jpg", width="500">
  <br>
  <i> Salvador Dali - Eucharistic Still Life (1952) </i>
</p>

## Status
Work in progress. Last updated: 5 September 2018.

### Table of contents

- [Tools, Techniques and Concepts](#tools-techniques-and-concepts)
- [Motivation And Project Description](#motivation-and-project-description)
- [Data Wrangling and Exploration](#data-wrangling-and-exploration)


## Tools, Techniques and Concepts

Python, Decision Tree, Random Forest, Cross Validation, Confusion Matrix

## Motivation And Project Description

In a [previous project](https://github.com/tommyzakhoo/epicurious-part-1), I cleaned and wrangled a set of data containing 15,709 recipes from the online recipe website Epicurious. Highly correlated variablles are also removed to help prevent [Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity).

In this project, I will continue digging into this dataset, using decision trees and random forests to build a classifier for predicting the ranking of recipes on Epicurious. This

## Data Wrangling and Exploration

I want to focus on binary classification with decision trees and random forests for this project. But ratings in the dataset goes from 0.0 to 5.0, in steps of 0.125. So, I am going to bin the ratings into two classes. First, let's take a look at the distribution of recipe ratings.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk/master/ratings.png", width="600">
</p>

It looks like recipe ratings tend to be high, with a median of 4.375. I am going to divide the ratings into two classes around the median: "bad" = strictly less thsn 4.375. "good" = equal to or greater than 4.375. I went through the ratings column and replaced all ratings that are "good" with a 1, and set the rest to 0.

The dataset that I am going to primarily work with, and has ratings binned into two class, can be found here: [recipes_data.csv](recipes_data.csv).

Note: at the time of writing, Epicurious' rating system has been changed to 0 to 4 instead. More details on the source of this dataset can be found in a [previous project](https://github.com/tommyzakhoo/epicurious-part-1).

## Decision Tree

## Random Forest

## Confusion Matrix, ROC, AUC

## Gini Coefficient

## Summary and Final Thoughts


