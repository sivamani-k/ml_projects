from textblob import TextBlob
text = "I love this product!"
sentiment = TextBlob(text).sentiment
print(sentiment)