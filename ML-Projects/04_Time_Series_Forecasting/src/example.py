import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
data = np.random.randn(100)
df = pd.DataFrame(data, columns=['value'])
model = ARIMA(df['value'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
print(forecast)