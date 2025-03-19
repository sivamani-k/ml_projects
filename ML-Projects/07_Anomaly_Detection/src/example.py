import numpy as np
from sklearn.ensemble import IsolationForest
data = np.random.randn(100, 2)
anomaly_detector = IsolationForest(contamination=0.1)
anomaly_detector.fit(data)
predictions = anomaly_detector.predict(data)
print(predictions)