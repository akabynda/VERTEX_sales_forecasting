import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings('ignore')


df = pd.read_csv('data.csv')

# Выбор нужных столбцов (Месяц и Продажи, уп)
df = df.iloc[:, [2, 3]]
df.columns = ['Месяц', 'Продажи, уп']

df['Месяц'] = pd.to_datetime(df['Месяц'], format='%m/%d/%Y')
df['Продажи, уп'] = pd.to_numeric(df['Продажи, уп'].replace(r'[^\d]', '', regex=True).astype(float))

df_aggr = df.copy()
df_aggr['Полугодие'] = df_aggr['Месяц'].dt.month.apply(lambda m: 1 if m <= 6 else 2)
df_aggr['Год'] = df_aggr['Месяц'].dt.year

# Группировка данных по году и полугодию, агрегация по сумме и среднему
grouped = df_aggr.groupby(['Год', 'Полугодие']).agg(
    Сумма=('Продажи, уп', 'sum'),
    Среднее=('Продажи, уп', 'mean')
).reset_index()

result_df = grouped[['Год', 'Полугодие', 'Среднее', 'Сумма']]
print(result_df.head())

result_df['Год'] = result_df['Год'].astype(int)
result_df['Полугодие'] = result_df['Полугодие'].astype(int)

X_sum = result_df[['Год', 'Полугодие', 'Среднее']]
y_sum = result_df['Сумма']

# Создание и обучение модели линейной регрессии для суммы продаж
model_sum = LinearRegression()
model_sum.fit(X_sum, y_sum)

# Предсказание суммы продаж для второго полугодия 2023 года
# Используем среднее значение продаж за 2023 год как признак
mean_sales_2023 = result_df[result_df['Год'] == 2023]['Среднее'].mean()
X_pred = pd.DataFrame({
    'Год': [2023],
    'Полугодие': [2],
    'Среднее': [mean_sales_2023]
})
prediction_sum = model_sum.predict(X_pred)


print(f"Предсказание суммарных продаж для второго полугодия 2023 года с учетом среднего: {prediction_sum[0]:.2f}")

X_sum = result_df[['Год', 'Полугодие']]
y_sum = result_df['Сумма']

# Создание и обучение модели линейной регрессии для суммы продаж
model_sum = LinearRegression()
model_sum.fit(X_sum, y_sum)

# Предсказание суммы продаж для второго полугодия 2023 года
# Используем среднее значение продаж за 2023 год как признак
X_pred = np.array([[2023, 2]])
prediction_sum = model_sum.predict(X_pred)

print(f"Предсказание суммарных продаж для второго полугодия 2023 года без учета среднего: {prediction_sum[0]:.2f}")

X_mean = result_df[['Год', 'Полугодие']]
y_mean = result_df['Среднее']
model_mean = LinearRegression()
model_mean.fit(X_mean, y_mean)

# Предсказание среднего значения продаж для второго полугодия 2023 года
prediction_mean = model_mean.predict(np.array([[2023, 2]]))

print(f"Предсказание среднего значения продаж для второго полугодия 2023 года: {prediction_mean[0]:.2f}")

df_monthly = df.resample('M', on='Месяц').sum()

# Вычисление значений ACF и PACF
acf_values = acf(df_monthly['Продажи, уп'], nlags=8)
pacf_values = pacf(df_monthly['Продажи, уп'], nlags=8)

# Вывод значений ACF и PACF
acf_df = pd.DataFrame({'Lag': range(len(acf_values)), 'ACF': acf_values})
pacf_df = pd.DataFrame({'Lag': range(len(pacf_values)), 'PACF': pacf_values})

print("Значения ACF:")
print(acf_df)
print("Значения PACF:")
print(pacf_df)

# Автоматический подбор параметров ARIMA
def auto_arima(data, max_p=5, max_q=5, d=1, seasonal=False, m=12):
    best_aic = np.inf
    best_order = None
    best_model = None

    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)
    for p in p_range:
        for q in q_range:
            try:
                if seasonal:
                    model = sm.tsa.statespace.SARIMAX(data, order=(p, d, q), seasonal_order=(p, d, q, m))
                else:
                    model = ARIMA(data, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
                    best_model = results
            except:
                continue
    return best_model, best_order

# Определение модели с учетом сезонности
model_auto_arima, best_order = auto_arima(df_monthly['Продажи, уп'], seasonal=False)

# Прогнозирование на следующие 6 месяцев
forecast = model_auto_arima.forecast(steps=6)

print(f"Лучший порядок ARIMA: {best_order}")
print(f"Предсказание ARIMA: {forecast.values}")
print(f"Предсказание ARIMA, сумма: {sum(forecast.values)}")
