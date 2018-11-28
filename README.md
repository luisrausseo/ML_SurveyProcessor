# Classify Customer's Survey Comments Using Machine Learning in Python

## The Problem

The survey's comments used in the context of this study, are the feedback from customers regarding the IT related services that they received. After their issue is solved, they are prompted to fill out a survey in which they can express their satisfaction.

These comments are then rated in a scale of -1 to 4. 

![alt text](https://github.com/luisrausseo/ML_SurveyProcessor/blob/master/Capture.PNG)

For the purpose of this study, the scores -1 and 0 are not considered, since the assigment of these values does not need a special algorithm.

- If it is negative, it will always be -1.
- If the comment is empty, it will be 0.

After receiving the survey's feedback from the customer, an agent in the quality team process it and give a subjective score to the customer comments. 

## The Data

There are XXXX entries in which the text from the comment is available, as well as its score in the format "__label__X" where X can be an integer from 1 to 4. 

| Label        | Text           |
| ------------- |--------------|
| __label__3      | Every single time I have a technical issue, and I contact you all, I receive immediate service!  It surprises me and pleases me every time.  From my experiences, using ITHelpcentral is easy, fast, and worry free.  Thank you! |
| __label__3      | I appreciate the walk-in service very much.  It is invaluable.      |
| __label__2 | amazing !      |

## Approach

One of the proposed solution of how to automate this process, is to use a machine learning algorithm to assign the scores. For this purpose, several Python machine learning algorithms are compared to define their performance in this particular problem. 

To handle text data, the bag of words models is used. To achieve that, TfidVectorizer is used to process the raw data.

```Python
from sklearn.feature_extraction.text import TfidfVectorizer
```
After processing the raw data, we split the data for out train and test set. The test set is 10% of the total data. Then we apply the machine learning algorithms to get the desired results.

### Naive Bayes classifier for multinomial models (MultinomialNB)

>[`MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB "sklearn.naive_bayes.MultinomialNB") implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification (where the data are typically represented as word vector counts, although tf-idf vectors are also known to work well in practice).

```Python
text_clf = Pipeline([('Tfidvect', TfidfVectorizer()), ('clf', MultinomialNB(fit_prior=False))])
text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print("Accuracy with NB = " + str(np.mean(predicted == y_test)))
```

#### Results

```Python
Accuracy with NB = 0.6606334841628959
```

![MultinomialNB Heatmap](https://github.com/luisrausseo/ML_SurveyProcessor/blob/master/results/M_NB.png)

|Label|Pedicted|Correct|Percent
|-|-|-|-|
|__label__1|61|44|72.1%
|__label__2|308|210|68.2%
|__label__3|47|16|34.0%
|__label__4|247|168|68.0%

# References
- [Python](https://www.python.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Numpy](http://www.numpy.org/)
- [Natural Language Toolkit](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Pandas](https://pandas.pydata.org/)
