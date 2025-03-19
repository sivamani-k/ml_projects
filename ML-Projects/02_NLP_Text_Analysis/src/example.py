from sklearn.feature_extraction.text import CountVectorizer
corpus = ["This is a sentence", "This is another sentence"]
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus)
print(x.toarray())