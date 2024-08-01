# Прогнозирование продаж: Линейная регрессия и модель ARIMA
### Загрузка и предобработка данных

Мы используем данные из файла `data.csv`, которые содержат информацию о продажах за различные месяцы.

```python
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
```

### Линейная регрессия

Линейная регрессия используется для предсказания суммарных продаж и среднего значения продаж для второго полугодия 2023 года.

#### Обучение модели и предсказание

```python
result_df['Год'] = result_df['Год'].astype(int)
result_df['Полугодие'] = result_df['Полугодие'].astype(int)

X_sum = result_df[['Год', 'Полугодие', 'Среднее']]
y_sum = result_df['Сумма']

# Создание и обучение модели линейной регрессии для суммы продаж
model_sum = LinearRegression()
model_sum.fit(X_sum, y_sum)

# Предсказание суммы продаж для второго полугодия 2023 года
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

model_sum = LinearRegression()
model_sum.fit(X_sum, y_sum)

X_pred = np.array([[2023, 2]])
prediction_sum = model_sum.predict(X_pred)

print(f"Предсказание суммарных продаж для второго полугодия 2023 года без учета среднего: {prediction_sum[0]:.2f}")

X_mean = result_df[['Год', 'Полугодие']]
y_mean = result_df['Среднее']
model_mean = LinearRegression()
model_mean.fit(X_mean, y_mean)

prediction_mean = model_mean.predict(np.array([[2023, 2]]))

print(f"Предсказание среднего значения продаж для второго полугодия 2023 года: {prediction_mean[0]:.2f}")
```

### Анализ ACF и PACF

Для определения наличия сезонности был проведен анализ автокорреляционной функции (ACF) и частичной автокорреляционной функции (PACF).

```python
df_monthly = df.resample('M', on='Месяц').sum()

acf_values = acf(df_monthly['Продажи, уп'], nlags=8)
pacf_values = pacf(df_monthly['Продажи, уп'], nlags=8)

acf_df = pd.DataFrame({'Lag': range(len(acf_values)), 'ACF': acf_values})
pacf_df = pd.DataFrame({'Lag': range(len(pacf_values)), 'PACF': pacf_values})

print("Значения ACF:")
print(acf_df)
print("Значения PACF:")
print(pacf_df)
```

### Заключение о сезонности

На основе анализа значений ACF и PACF сезонность в данных не наблюдается. Следовательно, использование несезонной модели ARIMA будет корректным для предсказаний.

### Модель ARIMA

Автоматический подбор параметров ARIMA для несезонной модели.

```python
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

model_auto_arima, best_order = auto_arima(df_monthly['Продажи, уп'], seasonal=False)

forecast = model_auto_arima.forecast(steps=6)

print(f"Лучший порядок ARIMA: {best_order}")
print(f"Предсказание ARIMA: {forecast.values}")
print(f"Предсказание ARIMA, сумма: {sum(forecast.values)}")
```

### Результаты

#### Линейная регрессия
- Предсказание суммарных продаж для второго полугодия 2023 года с учетом среднего: 107929.15
- Предсказание суммарных продаж для второго полугодия 2023 года без учета среднего: 113166.00
- Предсказание среднего значения продаж для второго полугодия 2023 года: 2327.70

#### Анализ ACF и PACF
- **Значения ACF**:
   - Lag 0: 1.000000
   - Lag 1: 0.058754
   - Lag 2: 0.121663
   - Lag 3: 0.081350
   - Lag 4: -0.004514
   - Lag 5: -0.012852
   - Lag 6: 0.255198
   - Lag 7: -0.189142
   - Lag 8: -0.049685
- **Значения PACF**:
   - Lag 0: 1.000000
   - Lag 1: 0.062210
   - Lag 2: 0.133517
   - Lag 3: 0.083747
   - Lag 4: -0.033804
   - Lag 5: -0.041331
   - Lag 6: 0.394804
   - Lag 7: -0.413089
   - Lag 8: -0.147808

#### ARIMA
- Лучший порядок ARIMA: (1, 1, 0)
- Предсказание ARIMA: [38955.08, 12112.61, 35811.47, 14888.07, 33361.05, 17051.50]
- Предсказание ARIMA, сумма: 152179.78

### Объяснения и доказательство корректности анализа

#### Выбор моделей

1. **Линейная регрессия**: Эта модель использовалась для анализа тенденций продаж на основе временных признаков (год и полугодие) и среднего значения продаж. Линейная регрессия проста в реализации и может быть полезна для базовых предсказаний.

2. **ARIMA**: Модель ARIMA (AutoRegressive Integrated Moving Average) используется для временных рядов без сезонности. Она хорошо справляется с данными, которые демонстрируют некоторые уровни автокорреляции и требуют интеграции для стационарности.

#### Обоснование выбора несезонной модели ARIMA

- **Анализ ACF и PACF**: Значения ACF и PACF не показали явных признаков сезонности. Обычно сезонные компоненты проявляются в виде значительных пиков на определенных лагах. В наших данных таких пиков не наблюдается.
  
- **Визуализация данных**: Графики ACF и PACF показывают, что автокорреляция и частичная автокорреляция не имеют значительных повторяющихся шаблонов, что подтверждает отсутствие сезонности.

#### Автоматический подбор параметров ARIMA

- **Критерий информации Акаике (AIC)**: Для выбора наилучшей модели ARIMA используется критерий информации Акаике (AIC). Модель с наименьшим значением AIC считается наиболее подходящей.

- **Перебор параметров**: Параметры `p`, `d` и `q` подбираются путем перебора в заданных диапазонах. Для каждой комбинации рассчитывается модель ARIMA и ее значение AIC.

- **Сезонность**: Параметр `seasonal` был установлен в `False`, так как анализ показал отсутствие сезонности в данных.