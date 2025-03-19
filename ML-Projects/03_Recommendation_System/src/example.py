import pandas as pd
from surprise import SVD, Dataset, Reader
ratings_dict = {'itemID': [1, 1, 1, 2, 2], 'userID': [1, 2, 3, 1, 2], 'rating': [5, 4, 3, 4, 5]}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)
prediction = algo.predict(1, 2)
print(prediction.est)