import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# === Загрузка данных ===
file_name = "Вся_работа.csv"
data = pd.read_csv(file_name, delimiter=",", encoding="utf-8")

file_name_test = "Запуск2.csv"
data_test = pd.read_csv(file_name_test, delimiter=",", encoding="utf-8")

# Используем только температуры T_ST_1...T_ST_12
temp_cols = [f"T_ST_{i}" for i in range(1, 13)]
train_data = data[temp_cols].fillna(0).to_numpy()
test_data = data_test[temp_cols].fillna(0).to_numpy()

# === Нормализация ===
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# === Автоэнкодер ===
input_dim = train_scaled.shape[1]
encoding_dim = 4  # латентное пространство (можно регулировать)

input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(8, activation="relu")(input_layer)
encoded = layers.Dense(encoding_dim, activation="relu")(encoded)

decoded = layers.Dense(8, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation=None)(decoded)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(optimizer="adam", loss="mse")

# === Обучение автоэнкодера ===
history = autoencoder.fit(
    train_scaled, train_scaled,
    epochs=30,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# === Ошибка реконструкции ===
reconstructions_train = autoencoder.predict(train_scaled, verbose=0)
mse_train = np.mean(np.square(train_scaled - reconstructions_train), axis=1)

reconstructions_test = autoencoder.predict(test_scaled, verbose=0)
mse_test = np.mean(np.square(test_scaled - reconstructions_test), axis=1)

# === Порог аномалии ===
threshold = mse_train.mean() + 3 * mse_train.std()

# === Генерация временного индекса для тестовых данных ===
start_time_test = datetime.strptime("10:50:30", "%H:%M:%S")
time_index_test = [start_time_test + timedelta(seconds=0.2 * i) for i in range(len(test_data))]

# # Создаём отображение уровня (горизонтальной линии) для каждого параметра по порядку в params
# levels_map = {p: i for i, p in enumerate(temp_cols)}
# # Подготовка словаря с масштабированными данными
# scaled_data = {}
# # Аналоговые параметры: нормализация ±10% и сдвиг на уровень
# for p in temp_cols:
#     if p in data_test.columns:
#         vals = pd.to_numeric(data_test[p], errors='coerce')
#         min_val, max_val = vals.min(), vals.max()
#         if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
#             # Если нет разброса — просто рисуем постоянную линию на уровне
#             norm = np.zeros(len(vals))
#         else:
#             pad = 0.1 * (max_val - min_val)
#             norm = (vals - (min_val - pad)) / ((max_val + pad) - (min_val - pad))
#         scaled_data[p] = norm.to_numpy() + levels_map[p]


# === Визуализация ===
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.6, 0.4],
    subplot_titles=("Температуры (T_ST_1...T_ST_12)", "Ошибка реконструкции (MSE)")
)

for p in temp_cols:
    # if p in scaled_data:
    fig.add_trace(
        go.Scatter(
            x=time_index_test,
            y=data_test[p],
            mode='lines',
            name=p,
            line_shape='linear',  # дискретные — ступенчатые линии
            customdata=data_test[p],
            hovertemplate=f"<b>{p}</b><br>Время: %{{x|%H:%M:%S}}<br>Значение: %{{customdata:.3f}}<extra></extra>"
        ),
        row = 1, col = 1
    )

# --- Нижний график (MSE + threshold) ---
fig.add_trace(
    go.Scatter(x=time_index_test, y=mse_test, mode='lines', name="MSE", line=dict(color='red')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=time_index_test, y=[threshold]*len(mse_test), mode='lines',
               name="Порог аномалии", line=dict(color='blue', dash='dash')),
    row=2, col=1
)

# --- Подсветка аномалий ---
anomaly_points = [i for i, val in enumerate(mse_test) if val > threshold]
if anomaly_points:
    fig.add_trace(
        go.Scatter(
            x=np.array(time_index_test)[anomaly_points],
            y=mse_test[anomaly_points],
            mode='markers',
            name="Аномалия",
            marker=dict(color='orange', size=8, symbol='x')
        ),
        row=2, col=1
    )
    # line=dict(color='red')

fig.update_layout(
    title="Анализ работы автоэнкодером",
    plot_bgcolor="#eeeeee",
    paper_bgcolor="#eeeeee",
    # xaxis_title="Время",
    # yaxis_title="Ошибка реконструкции (MSE)",
    # # template="plotly_dark",
    # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode = "x unified",
    margin = dict(l=80, r=50, t=80, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    # xaxis = dict(title="Время", showgrid=True, tickformat='%H:%M:%S')
)

# # Горизонтальные линии и подписи слева
# for p in temp_cols:
#     if p in scaled_data:
#         lvl = levels_map[p]
#         fig.add_hline(y=lvl, line=dict(color="black", width=1))
#         fig.add_annotation(xref='paper', x=-0.02, y=lvl + 0.5, text=p, showarrow=False, font=dict(size=20), yref='y')
fig.update_xaxes(tickformat="%H:%M:%S")

fig.show()
