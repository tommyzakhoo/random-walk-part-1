# A Random Walk Through The Forest
## Predicting Online Recipe Ratings Using A Decision Tree

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk/master/dail.jpg", width="500">
  <br>
  <i> Salvador Dali - Eucharistic Still Life (1952) </i>
</p>

## Status
Completed.

## Table of contents

- [Tools, Techniques and Concepts](#tools-techniques-and-concepts)
- [Motivation And Project Description](#motivation-and-project-description)
- [Data Wrangling and Exploration](#data-wrangling-and-exploration)
- [Decision Tree](#decision-tree)
- [Gini Impurity](#gini-impurity)
- [Tree Pruning](#tree-pruning)
- [Confusion Matrix](#confusion-matrix)
- [Feature Importance](#feature-importance)
- [Summary and Final Thoughts](#summary-and-final-thoughts)

## Tools, Techniques and Concepts

Python, Data Wrangling, Decision Tree, Gini Impurity, Tree Pruning, Confusion Matrix

## Motivation And Project Description

In a [previous project](https://github.com/tommyzakhoo/epicurious-part-1), I cleaned and wrangled a set of data containing 15,709 recipes from the online recipe website, Epicurious. Highly correlated variablles were also removed to help prevent [Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity).

In this project, I will continue digging into this dataset, building a decision tree classifier to try and predict the rating of recipes on Epicurious. This preliminary attempt will be refined in a later part of this project.

## Data Wrangling and Exploration

I want to focus on binary classification with decision trees for this project. But ratings in the dataset goes from 0.0 to 5.0, in steps of 0.125. So, I am going to bin the ratings into two classes. First, let's take a look at the distribution of recipe ratings.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk/master/ratings.png", width="600">
</p>

It looks like recipe ratings tend to be high, with a median of 4.375. I am going to divide the ratings into two classes around the median: "bad" = strictly less than 4.375. "good" = equal to or greater than 4.375. I went through the ratings column and replaced all ratings that are "good" with a 1, and set the rest to 0. Roughly 54% of the recipes has a "1", which is a nice balance.

The dataset that I am going to primarily work with, and has ratings binned into two class, can be found here: [recipes_data.csv](recipes_data.csv). 

Note that during the [previous project](https://github.com/tommyzakhoo/epicurious-part-1), in addition to being cleaned of possible errors, the data had columns "fat", "Portland", "non-alcoholic", "brunch" removed to avoid multicollinearity.

Also, at the time of writing, the Epicurious rating system appears to have been changed. Ratings seems to only range from 0 to 4. More details on the source of this dataset can be found in the [previous project](https://github.com/tommyzakhoo/epicurious-part-1).

## Decision Tree

I could use also use logistic regression to build a classifier, but I am going to try a non-linear method, by building a decision tree classifier instead. This will be done using sklearn, which comes with a [pretty good user guide](http://scikit-learn.org/stable/modules/tree.html).

In the [tips for practical use section](http://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use), sklearn's decision tree guide says to try a depth of 3 initially, then increase the depth. This is what I will be doing.

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

As I am using Windows 10, visualizing the decision tree with graphviz can be a little tricky. However, the "tree.txt" output can visualized with a website such as [www.webgraphviz.com](http://www.webgraphviz.com/).

The output with maximum tree depth set to 3 is shown below. X[i] refers to the i-th column of the feature dataset X in the input. What this decision tree is saying is that, starting from the top "root" node, and the data is divided into two sets based on whether X[60] <= 0.5 or X[60] > 0.5. 

Then, these two smaller dataset is further divided into respective halves using the decision rule at their respective node. This process is recursively applied until final level or leaf nodes are reached, and a label is recommended for the data in each leaf node.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/tree1.png">
</p>

"Samples" refer to how many data point is left at a node after the prior split. "Value" tells us the distributions of actual labels in that sample. E.g. a value of [10,20] means that 10 data points have the label "bad", while 20 data points are actually labeled "good".

## Gini Impurity

The [gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) is reported at each node of the tree above. For example, gini = 0.496 is reported at the root node in the tree above. Gini impurity is an interesting metric for how "mixed" a set of data is, using this process:

1) Pick a random data point from the set.
2) Randomly label the data point, using the proportions of the labels in the set as the probabilities.

In this case, I have two labels "good" and "bad" for recipe rankings. If the probability of picking a "good" recipe in a data set is p, then the gini impurity is simply p(1-p) + (1-p)p = 2p(1-p). This is maximized at p = 0.5, when the data is evenly split. This concept can be extended to sets with multiple labels by summing up p_i (1-p_i), where p_i is the proportion of the ith label in the set.

## Tree Pruning

There are better methods for depth selection, but here I will simply increase the depth of my decision tree until the leaves contain a low number of samples. I eventually settled on a depth of six.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/depth_six_tree.png">
</p>

At each node, there is a split, so there are 2 nodes after the first split, 4 nodes after the second split and so on. After split k, 2 to the power of k nodes are created. As seen in the figure above, this can get out of hand really quickly. I will be reducing the complexity of this tree by deleting nodes, which is often called "pruning".

There are [more sophisticated ways](https://en.wikipedia.org/wiki/Pruning_(decision_trees)) to prune decisions tree. I might try something better in a future project, but for now, I will be pruning the tree by hand. For example, I will merge two leaf nodes that are recommending the same label. And splits with too few samples might be ignored, unless they lead to great splits further down the tree. I will be on the lookout for great splits using the gini impurity as a guide.

The decision tree looks a lot more manageable after its haircut!

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/pruned.png">
</p>

## Confusion Matrix

As shown in the figure below, I evaluated my decision tree classifier using a confusion matrix.

<p align="left">
  <img src="https://raw.githubusercontent.com/tommyzakhoo/random-walk-part-1/master/confusion_matrix.png", width=600>
</p>

There are [many measures](https://en.wikipedia.org/wiki/Confusion_matrix) that can be computed from a confusion matrix. I will just focus on a few here.

The most striking thing about my classifier is that it is very good at classifying recipes with good ratings. The sensitivity or true positive rate is 7571/8566 = 88.384%. However, it is terrible at labeling recipes with bad ratings, with a specificity or true negative rate of just 22.27%!

The precision or positive predictive value is 7571/(5551+7571) = 57.696%. The negative predictive value is 1591/(1591+995) = 61.523%. These are a little bit better than the actual proportions of 54% good ratings and 45% bad ratings.

These poor results could simply mean that it is very difficult to tell recipes with good ratings apart from bad. It will be interesting to see if I can improve my classifer in part 2 of this project.

## Feature Importance

Despite how weak my classifier is, I could still try to learn about the features that affects recipe ratings, by looking at great splits in the pruned tree.

The first great split starts at the root node. Recipes with a 1 in the 60th column (X[60] > 0.5) tend to have good ratings, as shown by "value = [2930,4401]" in the node, which means 4401 recipes with good ratings are in the set, vs 2930 with bad ratings. X[60] is the variable "bon appétit", which refers to recipes that are from the cooking magazine "bon appétit".

In the next level, the set of recipes with calories X[0] > 225.5 contains 3423 good ratings vs 2029 bad ratings. Looks like recipes that are low in calories are disliked! This appears in another place on the tree, where X[0] <= 199.5, produces a set with 1269 bad ratings and 989 good ratings.

Another great classification of recipes with good rating comes from picking recipes with X[616] = 0, X[590] = 0, and X[450] = 0. These three variables are respectively: thanksgiving, stir-fry, peas. The set of recipes that are not for Thanksgiving, is not stir-fried, and does not contain peas, has 2851 good ratings vs 1687 bad ratings.

Based on my decision tree classifier, incorporating one or more of these might help your recipe get a higher rating on Epicurious, at least according to this dataset.

- Not from "bon appétit" magazine.
- High calories (yikes!).
- Not a Thanksgiving recipe.
- Is not stir-fried.
- Does not contain peas.

## Summary and Final Thoughts

Well, it turns out that I did not succeed at creating a great decision tree classifer for recipe ratings! Despite that, I am still excited by the various possibilities for improvement in part 2 of this project, and is very eager to see if they will work. 

Below is a summary of what I have done in this project.

- Wrangled a set of recipes data from a [previous project](https://github.com/tommyzakhoo/epicurious-part-1).
- Built a decision tree classifier for predicting if a recipe has a "good" >= 4.375 rating, or a "bad" < 4.375 rating.
- Explained gini impurity and pruned the decision tree manually.
- Evaluated the classifier with a confusion matrix.
- Calculated the sensitivity, specificity, precision, and negative predictive value.
- Used the decision tree to evaluate feature importance.
