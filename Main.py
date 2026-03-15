import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# === Загрузка данных ===
file_name = "Perfect_All_AnPar.csv"
data = pd.read_csv(file_name, delimiter=",")

# Определяем параметры
main_param = "evNvd"
exog_params = ["evPk", "evGt", "evNst", "evTin", "evTt"]

# Удалим пропуски, если есть
data = data[[main_param] + exog_params].dropna().reset_index(drop=True)

# Разделение на train/test
split_idx = int(len(data) * 0.95)

# Обучающие данные
train_y = data[main_param].iloc[:split_idx]
train_exog = data[exog_params].iloc[:split_idx]

# Тестовые данные (для прогноза)
test_y = data[main_param].iloc[split_idx:]
test_exog = data[exog_params].iloc[split_idx:]

# === Обучение модели на 95% ===
model = SARIMAX(train_y, exog=train_exog, order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# === Прогноз ===
forecast_res = model_fit.get_forecast(steps=len(test_y), exog=test_exog)
forecast_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int(alpha=0.05)

# === Построение графика ===
plt.figure(figsize=(14, 6))

# Отображаем только реальные значения
plt.plot(range(len(data)), data[main_param], label="Фактические значения", color='blue')

# Отображаем прогноз
forecast_range = range(split_idx, len(data))
plt.plot(forecast_range, forecast_mean, label="Прогноз ARIMAX", color='red')

# Доверительный интервал
plt.fill_between(forecast_range,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label="95% доверительный интервал")

# Вертикальная линия — начало прогноза
plt.axvline(x=split_idx, color='black', linestyle='--', label="Начало прогноза")

plt.title("ARIMAX-прогноз параметра evNvd (последние 5%)")
plt.xlabel("Индекс времени")
plt.ylabel(main_param)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
