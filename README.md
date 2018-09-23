# A Random Walk Through The Forest
## Predicting Online Recipe Ranking Using A Decision Tree

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk/master/dail.jpg", width="500">
  <br>
  <i> Salvador Dali - Eucharistic Still Life (1952) </i>
</p>

## Status
Work in progress. Last updated: 23 September 2018.

### Table of contents

- [Tools, Techniques and Concepts](#tools-techniques-and-concepts)
- [Motivation And Project Description](#motivation-and-project-description)
- [Data Wrangling and Exploration](#data-wrangling-and-exploration)
- [Decision Tree](#decision-tree)
- [Gini Impurity](#gini-impurity)
- [Tree Pruning](#tree-pruning)
- [Confusion Matrix)(#confusion-matrix)
- [Important Features](#important-features)
- [Summary and Final Thoughts](#summary-and-final-thoughts)

## Tools, Techniques and Concepts

Python, Cross Validation, Decision Tree, Gini Impurity, Pruning, Confusion Matrix

## Motivation And Project Description

In a [previous project](https://github.com/tommyzakhoo/epicurious-part-1), I cleaned and wrangled a set of data containing 15,709 recipes from the online recipe website Epicurious. Highly correlated variablles are also removed to help prevent [Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity).

In this project, I will continue digging into this dataset, using decision trees and random forests to build a classifier for predicting the ranking of recipes on Epicurious. This

## Data Wrangling and Exploration

I want to focus on binary classification with decision trees and random forests for this project. But ratings in the dataset goes from 0.0 to 5.0, in steps of 0.125. So, I am going to bin the ratings into two classes. First, let's take a look at the distribution of recipe ratings.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk/master/ratings.png", width="600">
</p>

It looks like recipe ratings tend to be high, with a median of 4.375. I am going to divide the ratings into two classes around the median: "bad" = strictly less than 4.375. "good" = equal to or greater than 4.375. I went through the ratings column and replaced all ratings that are "good" with a 1, and set the rest to 0. Roughly 54% of the recipes has a "1", which is a nice balance.

The dataset that I am going to primarily work with, and has ratings binned into two class, can be found here: [recipes_data.csv](recipes_data.csv). 

Note that during the [previous project](https://github.com/tommyzakhoo/epicurious-part-1), in addition to being cleaned of possible errors, the data had columns "fat", "Portland", "non-alcoholic", "brunch" removed to avoid multicollinearity.

Also, at the time of writing, Epicurious' rating system appears to have been changed. Ratings seems to only range from 0 to 4. More details on the source of this dataset can be found in a [previous project](https://github.com/tommyzakhoo/epicurious-part-1).

## Decision Tree

I could use logistic regression to build a classifier, but I am going to try a non-linear method, by building a decision tree classifier. This will be done using sklearn, which comes with a pretty good user guide](http://scikit-learn.org/stable/modules/tree.html).

In the [tips for practical use section](http://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use), the sklearn's decision tree guide says to try a depth of 3 initially, then increase the depth. This is what I will be doing.

```Python
import pandas as pd
import graphviz

from sklearn import tree
from sklearn.model_selection import train_test_split

data = pd.read_csv('recipes_data.csv') # read data from csv into dataframe

X = data.drop('rating', axis=1) # drop labels
X = X.iloc[:,1:] # drop recipe titles
Y = data['rating']  # labels to predict

clf = tree.DecisionTreeClassifier(max_depth=3) # starts with a depth 3 tree
clf = clf.fit(X, Y) # fit the tree

# save decision tree to text file
dot_data = tree.export_graphviz(clf, class_names=['bad','good'], out_file='tree.txt')
```

As I am using Windows 10, visualizing the decision tree with graphviz can be a little tricky. However, the "tree.txt" output can visualized with a website such as [www.webgraphviz.com](http://www.webgraphviz.com/)

The output with maximum tree depth set to 3 is shown below. X[i] refers to the i-th column of the feature dataset X in the input. What this decision tree is saying is that, starting from the top "root" node, and the data is divided into two sets based on whether X[60] <= 0.5 or X[60] > 0.5. 

Then, these two smaller dataset is further divided into respective halves using the decision rule at their respective node. This process is recursively applied until final level or leaf nodes are reached, and a label is recommended for the data in each leaf node.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/tree1.png">
</p>

"Samples" refer to how many data point is left at a node after the prior split. "Value" tells us the distributions of actual labels in that sample. E.g. a value of [10,20] means that 10 data points have the label "bad", while 20 data points are actually labeled "good".

## Gini Impurity

The [gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) is reported at each node of the tree above. For example, gini = 0.496 is reported at the root node in the tree above. gini impurity is an interesting metric for how "mixed" a set of data is, using this process:

1) Pick a random data point from the set.
2) Randomly label the data point, using the proportions of the labels in the set as the probabilities.

In this case, I have two labels "good" and "bad" for recipe rankings. If the probability of picking a "good" recipe in a data set is p, then the gini impurity is simply p(1-p) + (1-p)p = 2p(1-p). This is maximized at p = 0.5, when the data is evenly split. This concept can be extended to sets with multiple labels by summing up p_i (1-p_i), where p_i is the proportion of the ith label in the set.

## Tree Pruning

There a better methods for depth selection, but here I simply increased the depth of my decision tree until the leaves contain a low number of samples, and eventually settled at a depth of six.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/depth_six_tree.png">
</p>

At each node, there is a split, so there are 2 nodes after the first split, 4 nodes after the second split and so on. After split k, 2 to the power of k nodes are created. As seen in the figure above, this can get out of hand really quickly. I will be reducing the complexity of this tree by deleting nodes, which is often called "pruning".

There are [more sophisticated ways](https://en.wikipedia.org/wiki/Pruning_(decision_trees)) to prune decisions tree. I might try something better in a future project, but for now, I will be pruning the tree by hand. For example, I will merge two leaf nodes recommending the same label, and splits with too few samples might be ignored unless it leads to a great split further down the tree. I will also look for great splits using the gini impurity.

The decision tree looks a lot more manageable after its haircut!

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/pruned.png">
</p>

## Confusion Matrix

As shown in the figure below, I evaluated my decision tree classifier using a confusion matrix.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/confusion%20matrix.png">
</p>

There are [many measures](https://en.wikipedia.org/wiki/Confusion_matrix) that can be computed from a confusion matrix. I will calculate a few here.

The most striking thing about my classifier is that it is very good at classifying recipes with good ratings. The sensitivity or true positive rate is 7571/8566 = 88.384%. However, it is terrible at labeling recipes with bad ratings, with a specificity or true negative rate of just 22.27%!

My interpretation of these results: there are specific things that will help a recipe get a high rating, but 

## Important Features

Despite the very low specificity of my classifier, I could still learn a lot about the features that push up recipe ratings, due to the very high sensitivity, by looking at great splits in the pruned tree.

The first great split starts at the root node. Recipes with a 1 in the 60th column (X[60] > 0.5) tend to have good ratings, as shown by "value = [2930,4401]" in the node, which means 4401 recipes with good ratings are in the set, vs 2930 with bad ratings.


## Summary and Final Thoughts

Below is a summary of what I have done in this project.

- Wrangled the cleaned set of recipes data from a [previous project] (https://github.com/tommyzakhoo/epicurious-part-1).
- Built a decision tree classifier for predicting if a recipe has a "good" >= 4.375 rating, or a "bad".
- Explained gini impurity and pruned the decision tree manually.
- Evaluated the classifier with a confusion matrix, obtained a true positive rate of 88.384%.
- Used the decision tree to discover which features 

