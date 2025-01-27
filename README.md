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

```
Accuracy with NB = 0.6606334841628959
```

![MultinomialNB Heatmap](https://github.com/luisrausseo/ML_SurveyProcessor/blob/master/results/M_NB.png)

|Label|Pedicted|Correct|Percent
|:-:|:-:|:-:|:-:|
|__label__1|61|44|72.1%
|__label__2|308|210|68.2%
|__label__3|47|16|34.0%
|__label__4|247|168|68.0%

### Stochastic Gradient Descent

>**Stochastic Gradient Descent (SGD)** is a simple yet very efficient approach to discriminative learning of linear classifiers under convex loss functions such as (linear) [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) and [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression). Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.

```Python
text_clf_svm = Pipeline([('Tfidvect', TfidfVectorizer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=45, shuffle=True))])
text_clf_svm = text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_test)
print("Accuracy with SVM = " + str(np.mean(predicted_svm == y_test)))
```

#### Results

```
Accuracy with SVM = 0.665158371040724
```

![SVM Heatmap](https://github.com/luisrausseo/ML_SurveyProcessor/blob/master/results/M_NB.png)

|Label|Pedicted|Correct|Percent
|:-:|:-:|:-:|:-:|
|__label__1|56|37|66.1%
|__label__2|352|227|64.5%
|__label__3|10|4|40.0%
|__label__4|245|173|70.6%

### Linear Support Vector Machine

>[`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC "sklearn.svm.LinearSVC") is another implementation of Support Vector Classification for the case of a linear kernel.

```Python
text_clf_lscv = Pipeline([('Tfidvect', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf_lscv = text_clf_lscv.fit(X_train, y_train)
predicted_lscv = text_clf_lscv.predict(X_test)
print("Accuracy with LSCV = " + str(np.mean(predicted_lscv == y_test)))
```

#### Results

```
Accuracy with LSCV = 0.6998491704374057
```

![LSCV Heatmap](https://github.com/luisrausseo/ML_SurveyProcessor/blob/master/results/LSVC.png)

|Label|Pedicted|Correct|Percent
|:-:|:-:|:-:|:-:|
|__label__1|73|47|64.4%
|__label__2|317|219|69.1%
|__label__3|71|30|42.3%
|__label__4|202|168|83.2%

### Natural Language Toolkit

```Python
stemmer = SnowballStemmer("english", ignore_stopwords=True)
       
class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
	    analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
            
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                             ('mnb', MultinomialNB(fit_prior=False))])
text_mnb_stemmed = text_mnb_stemmed.fit(X_train, y_train)
predicted_mnb_stemmed = text_mnb_stemmed.predict(X_test)
print("Accuracy with nltk: " + str(np.mean(predicted_mnb_stemmed == y_test)))
```

#### Results

```
Accuracy with nltk: 0.6591251885369532
```

![NLKT Heatmap](https://github.com/luisrausseo/ML_SurveyProcessor/blob/master/results/NLKT.png)

|Label|Pedicted|Correct|Percent
|:-:|:-:|:-:|:-:|
|__label__1|82|51|62.2%
|__label__2|299|207|69.2%
|__label__3|83|27|32.5%
|__label__4|199|152|76.4%

## Observations

- Visually the LinearSVC model shows better accuracy overall for the four labels. 
- There is a diffused area around the center of the graph: __label__2 and __label__3. This is caused by the subjectivity of the score: what an agent might consider a score of 3, another one can decide that is a 2. If we consider both labels in labels 2 and 3 for the LinearSVC model, the percentage significantly increases.

|Label|Pedicted|Correct|Percent
|:-:|:-:|:-:|:-:|
|__label__1|73|47|64.4%
|__label__2|317|276|87.1%
|__label__3|71|56|78.9%
|__label__4|202|168|83.2%

## Pending

- Attempt to clean further the data.
- Review discrepancies between labels 2 and 3.
- Seek other models to compare with. 

## References
- [Python](https://www.python.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Numpy](http://www.numpy.org/)
- [Natural Language Toolkit](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Pandas](https://pandas.pydata.org/)
