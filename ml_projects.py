import os

def create_project_structure(base_path="ML-Projects"):
    projects = [
        "01_Image_Classification",
        "02_NLP_Text_Analysis",
        "03_Recommendation_System",
        "04_Time_Series_Forecasting",
        "05_Generative_Adversarial_Networks",
        "06_Object_Detection",
        "07_Anomaly_Detection",
        "08_Sentiment_Analysis",
        "09_Reinforcement_Learning",
        "10_AutoML_Pipeline"
    ]
    
    os.makedirs(base_path, exist_ok=True)
    
    for project in projects:
        project_path = os.path.join(base_path, project)
        os.makedirs(project_path, exist_ok=True)
        
        # Create necessary directories
        os.makedirs(os.path.join(project_path, "notebooks"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "src"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "data"), exist_ok=True)

        # Create README.md
        with open(os.path.join(project_path, "README.md"), "w") as f:
            f.write(f"# {project.replace('_', ' ')}\n\nDescription and details coming soon.")

        # Create requirements.txt
        with open(os.path.join(project_path, "requirements.txt"), "w") as f:
            f.write("# Add dependencies here\n")
    
    # Example scripts for each project
    example_code = {
        "01_Image_Classification": """import tensorflow as tf
from tensorflow import keras
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)""",
        
        "02_NLP_Text_Analysis": """from sklearn.feature_extraction.text import CountVectorizer
corpus = ["This is a sentence", "This is another sentence"]
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus)
print(x.toarray())""",
        
        "03_Recommendation_System": """import pandas as pd
from surprise import SVD, Dataset, Reader
ratings_dict = {'itemID': [1, 1, 1, 2, 2], 'userID': [1, 2, 3, 1, 2], 'rating': [5, 4, 3, 4, 5]}
df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)
prediction = algo.predict(1, 2)
print(prediction.est)""",
        
        "04_Time_Series_Forecasting": """import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
data = np.random.randn(100)
df = pd.DataFrame(data, columns=['value'])
model = ARIMA(df['value'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
print(forecast)""",
        
        "05_Generative_Adversarial_Networks": """import tensorflow as tf
generator = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)), tf.keras.layers.Dense(784, activation='sigmoid')])
discriminator = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)), tf.keras.layers.Dense(1, activation='sigmoid')])""",
        
        "06_Object_Detection": """import cv2
import numpy as np
image = np.zeros((500, 500, 3), dtype='uint8')
cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), 2)
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()""",
        
        "07_Anomaly_Detection": """import numpy as np
from sklearn.ensemble import IsolationForest
data = np.random.randn(100, 2)
anomaly_detector = IsolationForest(contamination=0.1)
anomaly_detector.fit(data)
predictions = anomaly_detector.predict(data)
print(predictions)""",
        
        "08_Sentiment_Analysis": """from textblob import TextBlob
text = "I love this product!"
sentiment = TextBlob(text).sentiment
print(sentiment)""",
        
        "09_Reinforcement_Learning": """import gym
env = gym.make("CartPole-v1")
state = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    if done:
        break
env.close()""",
        
        "10_AutoML_Pipeline": """from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))"""
    }

    # Ensure src directory exists before writing example.py
    for project, code in example_code.items():
        src_path = os.path.join(base_path, project, "src")
        os.makedirs(src_path, exist_ok=True)  # Fix: Create src folder before writing

        with open(os.path.join(src_path, "example.py"), "w") as f:
            f.write(code)

    # Create main README file
    with open(os.path.join(base_path, "README.md"), "w") as f:
        f.write("# 10 Most Wanted Machine Learning Projects\n\n")
        f.write("This repository contains 10 high-impact machine learning projects with code, datasets, and notebooks.\n\n")
        f.write("## Projects\n\n")
        for project in projects:
            f.write(f"- [{project.replace('_', ' ')}](./{project})\n")
    
    print(f"✅ Project structure created at {base_path}")

# Run the function to create the project structure
create_project_structure()
