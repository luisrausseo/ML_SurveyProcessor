import pandas as pd
import numpy as np
import pymssql

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

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
		
	def PipelineLSCV(self, debug):
		text_clf_lscv = Pipeline([('Tfidvect', TfidfVectorizer()), ('clf', LinearSVC())])
		text_clf_lscv = text_clf_lscv.fit(self.X_train, self.y_train)
		predicted_lscv = text_clf_lscv.predict(self.X_test)
		if (debug == True):
			print("Accuracy with LSCV = " + str(np.mean(predicted_lscv == self.y_test)))
		return text_clf_lscv
		
	def getData(self):
		server = "Kadett.ttu.edu"
		user = "hc_webapps"
		password = "Uf9sOsUw"
		dataBase = "Footprints"
		
		conn = pymssql.connect(server, user, password, dataBase)
		cursor = conn.cursor()
		
		sql = "";
		
		cursor.execute("SELECT CONCAT('__label__', label) as label, LTRIM(SUBSTRING([text], CHARINDEX('>', [text])+5, 1000)) as text FROM (
	  SELECT LTRIM(RTRIM(SUBSTRING([mrALLDESCRIPTIONS], 0, CHARINDEX('ORIGINAL ISSUE =', [mrALLDESCRIPTIONS])))) as text 
	  ,[Customer__bComments__bRating] as label
	  FROM [Footprints].[dbo].[MASTER14] 
	  WHERE NOT [Customer__bComments__bRating] IN (0, '__u1')
	 ) a'")
		
		conn.close()

###############################################
dataPath = "data\survey14DBV2.csv"
Survey13 = SurveyAgent(dataPath)
model = Survey13.PipelineLSCV(False)
model = Survey13.getData()
# print(model.predict(["Paul (help line) and Josh (walk in) both were very reassuring , knowledgeable. Thank you so much!"]))