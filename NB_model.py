import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        #
        self.df['category_id'] = self.df['label'].factorize(sort=True)[0]
        self.category_id_df = self.df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
        self.category_to_id = dict(self.category_id_df.values)
        self.labels = self.df.category_id
        id_to_category = dict(self.category_id_df[['category_id', 'label']].values)
        self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test = train_test_split(self.df.text, self.df.label, self.df.index, test_size=0.10, random_state=45)
        self.df.head()

    def graphClasses(self):
        fig = plt.figure(figsize=(8,6))
        self.df.groupby('label').text.count().plot.bar(ylim=0)
        plt.show()

    def getNgrams(self, N):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')
        self.features = self.tfidf.fit_transform(self.df.text).toarray()
        self.labels = self.df.category_id
        self.features.shape
        for label, category_id in sorted(self.category_to_id.items()):
            features_chi2 = chi2(self.features, self.labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(self.tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(label))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
            print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
        

    def PipelineNB(self):
        text_clf = Pipeline([('Tfidvect', TfidfVectorizer()), ('clf', MultinomialNB(fit_prior=False))])
        text_clf = text_clf.fit(self.X_train, self.y_train)
        predicted = text_clf.predict(self.X_test)
        print("Accuracy with NB = " + str(np.mean(predicted == self.y_test)))
        return text_clf

    def PipelineSVM(self):
        text_clf_svm = Pipeline([('Tfidvect', TfidfVectorizer()),
                                 ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=45, shuffle=True))])
        text_clf_svm = text_clf_svm.fit(self.X_train, self.y_train)
        predicted_svm = text_clf_svm.predict(self.X_test)
        print("Accuracy with SVM = " + str(np.mean(predicted_svm == self.y_test)))
        return text_clf_svm

    def PipelineLSCV(self):
        text_clf_lscv = Pipeline([('Tfidvect', TfidfVectorizer()), ('clf', LinearSVC())])
        text_clf_lscv = text_clf_lscv.fit(self.X_train, self.y_train)
        predicted_lscv = text_clf_lscv.predict(self.X_test)
        print("Accuracy with LSCV = " + str(np.mean(predicted_lscv == self.y_test)))
        return text_clf_lscv

    def gridSearchNB(self):
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
        gs_clf = GridSearchCV(self.text_clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(self.X_train, self.y_train)
        print(gs_clf.best_score_)
        print(gs_clf.best_params_)

    def gridSearchSVM(self):
        parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}
        gs_clf_svm = GridSearchCV(self.text_clf_svm, parameters_svm, n_jobs=-1)
        gs_clf_svm = gs_clf_svm.fit(self.X_train, self.y_train)
        print(gs_clf_svm.best_score_)
        print(gs_clf_svm.best_params_)

    def nltkModel(self):
        ##nltk.download()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        
        class StemmedCountVectorizer(CountVectorizer):
            def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
            
        stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

        text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                                     ('mnb', MultinomialNB(fit_prior=False))])

        text_mnb_stemmed = text_mnb_stemmed.fit(self.X_train, self.y_train)

        predicted_mnb_stemmed = text_mnb_stemmed.predict(self.X_test)

        print("Accuacy with nltk: " + str(np.mean(predicted_mnb_stemmed == self.y_test)))
        return text_mnb_stemmed

    def graphHeatMap(self, model):
        print("Generating graph")
        model_pred = model.predict(self.X_test)
        conf_mat = confusion_matrix(self.y_test, model_pred)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=self.category_id_df.label.values, yticklabels=self.category_id_df.label.values)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        

###############################################
dataPath = "data\survey14DBV2.csv"
Survey13 = SurveyAgent(dataPath)
# Survey13.graphClasses()
##Survey13.getNgrams(10)
# Survey13.graphHeatMap(Survey13.PipelineNB())
Survey13.graphHeatMap(Survey13.PipelineSVM())
##Survey13.graphHeatMap(Survey13.PipelineLSCV())
##Survey13.gridSearchNB()
##Survey13.gridSearchSVM()
##Survey13.graphHeatMap(Survey13.nltkModel())
