import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np

from sklearn import svm

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

class SurveyAgent:
    def __init__(self, dataPath):
        self.df = pd.read_csv(dataPath, encoding='utf-8')
        col = ['label', 'text']
        self.df = self.df[col]
        self.df = self.df[pd.notnull(self.df['text'])]
        self.df.columns = ['label', 'text']
        #
        self.df.text = self.df.text.apply(lambda x: x.strip().encode().decode())
        self.df = self.df[~self.df.text.str.contains("duplicate")]
        #self.df.text = self.df.text.apply(lambda x: x[0:x.find("Original Issue = ")])
        #self.df = self.df[self.df.label != '__label__0']
        #
        self.df['category_id'] = self.df['label'].factorize(sort=True)[0]
        self.category_id_df = self.df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(self.category_id_df.values)
        id_to_category = dict(self.category_id_df[['category_id', 'label']].values)
        self.df.head()


    def graphClasses(self):
        fig = plt.figure(figsize=(8,6))
        self.df.groupby('label').text.count().plot.bar(ylim=0)
        plt.show()


    def processData(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
        self.features = self.tfidf.fit_transform(self.df.text).toarray()
        self.labels = self.df.category_id
        self.features.shape


    def getNgrams(self):
        N = 10
        for label, category_id in sorted(self.category_to_id.items()):
            features_chi2 = chi2(self.features, self.labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(self.tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(label))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
            print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


    def modelFit(self):
        print("Fitting Model")
        self.model = LinearSVC()
        self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test = train_test_split(self.features, self.labels, self.df.index, test_size=0.10, random_state=0)
        self.model.fit(self.X_train, self.y_train)


    def graphHeatMap(self):
        print("Generating graph")
        self.y_pred = self.model.predict(self.X_test)
        conf_mat = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=self.category_id_df.label.values, yticklabels=self.category_id_df.label.values)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def modelCompare(self):
        models = [
            RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state=0),
        ]
        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
          model_name = model.__class__.__name__
          accuracies = cross_val_score(model, self.features, self.labels, scoring='accuracy', cv=CV)
          for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

        sns.boxplot(x='model_name', y='accuracy', data=cv_df)
        sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.show()


###############################################
dataPath = "data\survey14DBV2.csv"
Survey13 = SurveyAgent(dataPath)
##Survey13.graphClasses()
Survey13.processData()
##Survey13.getNgrams()
Survey13.modelFit()
Survey13.graphHeatMap()
##Survey13.modelCompare()
