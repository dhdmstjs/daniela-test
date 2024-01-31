import pandas as pd 
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot



df = pd.read_csv('Jong_1D.csv')
df.columns = ['ds', 'y']
df.head()
df.tail()

# df['cap'] = 8000
# df['floor'] = 3000

df_prophet = Prophet(growth='flat' ,changepoint_prior_scale=0.9, changepoint_range=0.9, yearly_seasonality=5, weekly_seasonality=5, seasonality_mode='additive'\
                     , holidays_prior_scale=20)
df_prophet.add_country_holidays(country_name='KR')
df_prophet.fit(df)



fcast_time = 30 # 1 year:365
df_forecast = df_prophet.make_future_dataframe(periods=fcast_time,freq='d')
df_forecast.tail(10)

df_forecast = df_prophet.predict(df_forecast)
df_forecast.tail()

fig, ax = plt.subplots(figsize=(16,5))
df_prophet.plot(df_forecast, ax=ax)
plt.show()



df_prophet.plot(df_forecast, xlabel='Date', ylabel='EC')
plt.show()

fig2 = df_prophet.plot_components(df_forecast)
plt.show()
