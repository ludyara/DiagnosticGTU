import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from Tun import Lin_Pm_MNRG1, Lin_Pm_MNRG2, Lin_Pm_MNRGa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Функция парсинга времени события в datetime (с той же базовой датой, что и start_time)
def parse_event_datetime_to_base(s, base_dt):
    if pd.isna(s):
        return None
    s = str(s).strip()
    # Иногда в поле может попадать дата и время в одном поле, либо используется запятая как разделитель
    # Уберём возможные даты и возьмём последний токен
    if ' ' in s:
        token = s.split()[-1]
    elif ',' in s and s.count(':') == 2:
        # если формат '10:50:30.26' — оставляем, если '24.10.2024,10:50:30.26' — берём часть после запятой
        parts = s.split(',')
        token = parts[-1]
    else:
        token = s
    token = token.replace(',', '.')
    try:
        dt = datetime.strptime(token, '%H:%M:%S.%f')
    except ValueError:
        try:
            dt = datetime.strptime(token, '%H:%M:%S')
        except ValueError:
            return None
    # Подставим ту же дату, что и у start_time (чтобы можно было сравнивать)
    return base_dt.replace(hour=dt.hour, minute=dt.minute, second=dt.second, microsecond=dt.microsecond)

def work_nasos(nas, Pm_ac, Pm_ai, Pm_im):
    # Настройки модели
    delay = 1.0  # задержка (сек)
    tau = 0.85  # постоянная времени для апериодического нарастания
    dt = 1.0  # шаг дискретизации (сек) — подгони под твои данные
    zeta = 0.5  # коэффициент затухания (чем меньше — тем больше перерегулирование)
    omega_n = 2.0  # собственная частота колебаний (рад/с)
    mnrg_signal = nas.astype(int)  # булевый сигнал (1 = насос включен)

    # Подготовка
    n = len(Pm_ac)
    pm_raw = np.zeros(n, dtype=float)

    delay_steps = int(round(delay / dt))

    # начальное условие можно взять 0 или текущее реальное значение
    y = 0.0
    for i in range(n):
        # вход с учётом задержки:
        if i - delay_steps >= 0:
            u = Pm_ac[i - delay_steps]
        else:
            u = 0.0

        # (дискретная интеграция первого порядка) — работает и при вкл., и при выкл.
        if mnrg_signal[i] == 1:
            y = y + (dt / tau) * (u - y)
        else:
            y = y + (dt / 3.0) * (u - y)
        pm_raw[i] = y

    # Определяем диапазон с запасом ±10%
    min_val = np.min(pd.to_numeric(data[Pm_ai], errors='coerce'))
    max_val = np.max(pd.to_numeric(data[Pm_ai], errors='coerce'))
    range_val = max_val - min_val
    min_adj = min_val - 0.1 * range_val
    max_adj = max_val + 0.1 * range_val

    # Нормализуем в диапазон 0..1
    pm_norm = (pm_raw - min_adj) / (max_adj - min_adj)

    # Сохраняем для отображения: нормализованная серия + сдвиг на уровень Pm_inR
    scaled_data[Pm_im] = pm_norm + levels_map[Pm_ai]

    # Также сохраним необработанный параметр для hover-подсказки
    analog_raw[Pm_im] = pm_raw

def grafic_imit(P_imit):
    if P_imit in scaled_data and P_imit in analog_raw:
        fig.add_trace(go.Scatter(
            x=time_index,
            y=scaled_data[P_imit],  # нормализованное значение для оси Y
            mode='lines',
            name="Oil pressure at the gearbox inlet (model)",
            line_shape='linear',  # если дискреты, ступеньками; для аналога — 'linear'
            line=dict(color='red', dash='dash'),
            hovertemplate=(
                    "<br>" +
                    # "Норм.: %{y:.3f}<br>" +  # нормализованное (отображаемое на графике)
                    "Значение: %{customdata:.3f}"  # исходное (для подсказки)
            ),
            customdata=np.array(analog_raw[P_imit])  # сырые данные
        ))

def train_simple_nn(data: pd.DataFrame, input_params: list, target_param: str, epochs: int = 50, batch_size: int = 32):
    """
    Универсальная функция для обучения простой нейросети.

    :param data: DataFrame с параметрами
    :param input_params: список колонок, которые идут на вход
    :param target_param: имя колонки, которую предсказываем
    :param epochs: количество эпох обучения
    :param batch_size: размер батча
    :return: (model, history, scaler_X, scaler_y)
    """
    # Отбираем данные
    X = data[input_params].copy()
    y = data[target_param].copy()

    # Заполняем NaN нулями (или можно сделать другой препроцессинг)
    X = X.fillna(0).to_numpy(dtype=float)
    y = y.fillna(0).to_numpy(dtype=float).reshape(-1, 1)

    # Масштабируем признаки и цель
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Разбиваем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.01, random_state=42)

    # Строим модель
    model = keras.Sequential([
        layers.Input(shape=(len(input_params),)),
        layers.Dense(10, activation="relu"),
        layers.Dense(1)  # регрессия
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Обучаем
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history, scaler_X, scaler_y

# === Загрузка аналоговых данных ===
file_name = "Запуск.csv"
data = pd.read_csv(file_name, delimiter=",", encoding="utf-8")

# Генерация времени вместо индекса (каждый индекс = +1 секунда)
start_time = datetime.strptime("10:50:30", "%H:%M:%S")
time_index = [start_time + timedelta(seconds=i) for i in range(len(data))]

# === Загрузка дискретных событий ===
events_file = "events.csv"
events = pd.read_csv(events_file, delimiter=",", encoding="cp1251", on_bad_lines="skip", engine="python")

# Очистим названия колонок от лишних пробелов / BOM
events.columns = events.columns.str.strip().str.replace('\ufeff', '')

# Приведём строковые поля к нормальному виду
for col in ['Время', 'Статус', 'Сообщение']:
    if col in events.columns:
        events[col] = events[col].astype(str).str.strip()

# Оставляем только статусы "Пришло" и "Снялось"
valid_statuses = set(['Пришло', 'Снялось'])
if 'Статус' in events.columns:
    events = events[events['Статус'].isin(valid_statuses)].copy()


# Добавим колонку event_dt (datetime) и отфильтруем непарсимые
if 'Время' in events.columns:
    events['event_dt'] = events['Время'].apply(lambda s: parse_event_datetime_to_base(s, start_time))
    events = events.dropna(subset=['event_dt']).copy()

# Получаем список параметров — можно редактировать
params = ["Pm_inR", "Pm_outNRG1", "Pm_outNRG2", "Pm_outNRGA", "МНРГа включить", "МНРГ1 включить", "МНРГ2 включить"]

# Создаём отображение уровня (горизонтальной линии) для каждого параметра по порядку в params
levels_map = {p: i for i, p in enumerate(params)}

# Подготовка словаря с масштабированными данными
scaled_data = {}

# Аналоговые параметры: нормализация ±10% и сдвиг на уровень
for p in params:
    if p in data.columns:
        vals = pd.to_numeric(data[p], errors='coerce')
        min_val = vals.min()
        max_val = vals.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            # Если нет разброса — просто рисуем постоянную линию на уровне
            norm = np.zeros(len(vals))
        else:
            pad = 0.1 * (max_val - min_val)
            norm = (vals - (min_val - pad)) / ((max_val + pad) - (min_val - pad))
        scaled_data[p] = norm.to_numpy() + levels_map[p]

# Дискретные параметры: строим серии 0/1 по времени (и нормализация ±10% чтобы уровни были видны)
unique_messages = []
if 'Сообщение' in events.columns:
    unique_messages = events['Сообщение'].astype(str).str.strip().unique()

# Словарь для хранения «сырых» 0/1 (нужен для hover) и нормализованных данных
discrete_raw = {}
analog_raw = {}
for p in params:
    if p in unique_messages:
        series = np.zeros(len(time_index), dtype=float)

        # берём только события с данным сообщением, сортируем по точному времени
        rows = events[events['Сообщение'].astype(str).str.strip() == p].copy()
        rows = rows.sort_values('event_dt')

        for _, row in rows.iterrows():
            evt_dt = row.get('event_dt', None)
            if evt_dt is None:
                continue
            # округляем вниз до секунды (floor) — так как временная ось у нас с шагом 1 сек
            evt_floor = evt_dt.replace(microsecond=0)
            idx = int((evt_floor - start_time).total_seconds())
            if idx < 0 or idx >= len(time_index):
                continue
            status = str(row.get('Статус', '')).strip()
            if status == 'Пришло':
                series[idx:] = 1.0
            elif status == 'Снялось':
                series[idx:] = 0.0

        # Сохраним «сырое» значение для hover (0/1)
        discrete_raw[p] = series

        # Нормализация для дискретов аналогично аналоговым: добавляем ±10% паддинг
        # Для дискретов min=0, max=1 => pad = 0.1
        pad = 0.1 * (1.0 - 0.0)
        denom = (1.0 + pad) - (0.0 - pad)   # = 1.0 + 2*pad
        scaled = (series - (0.0 - pad)) / denom
        # сдвигаем на уровень параметра (levels_map должен быть объявлен ранее)
        scaled_data[p] = scaled + levels_map[p]

# === Формирование имитации Pm_inR (универсальный способ) ===
# Дискреты из "сырых" данных
mnrga = discrete_raw.get("МНРГа включить", np.zeros(len(time_index)))
mnrg1 = discrete_raw.get("МНРГ1 включить", np.zeros(len(time_index)))
mnrg2 = discrete_raw.get("МНРГ2 включить", np.zeros(len(time_index)))
N_MNRG1 = pd.to_numeric(data["N_MNRG1"], errors='coerce').to_numpy()
N_MNRG2 = pd.to_numeric(data["N_MNRG2"], errors='coerce').to_numpy()
N_MNRGa = pd.to_numeric(data["N_MNRGa"], errors='coerce').to_numpy()
Pm_outNRG1 = pd.to_numeric(data["Pm_outNRG1"], errors='coerce').to_numpy()
Pm_outNRG2 = pd.to_numeric(data["Pm_outNRG2"], errors='coerce').to_numpy()
Pm_outNRGA = pd.to_numeric(data["Pm_outNRGA"], errors='coerce').to_numpy()
mnrg_sum = mnrga + mnrg1 + mnrg2
# Пример произвольной формулы (можно менять)
Pm_1 = mnrg2 * mnrg1 * N_MNRG2 / 2678.4 + mnrg1 * N_MNRG1 / 239.0 + mnrg1 * mnrga * N_MNRGa / 2678.4
Pm_2 = mnrg2 * mnrg1 * N_MNRG1 / 2678.4 + mnrg2 * N_MNRG2 / 239.0 + mnrg2 * mnrga * N_MNRGa / 2678.4
Pm_a = mnrga * mnrg1 * N_MNRG1 / 2678.4 + mnrga * N_MNRGa / 239.0 + mnrg2 * mnrga * N_MNRG2 / 2678.4

mask = (mnrg1 > 0) | (mnrg2 > 0) | (mnrga > 0)  # булев массив
mask_int = mask.astype(int)                     # 0/1

# Поэлементное условие: если mnrg_sum > 1, делим на 2
Pm_inR = np.where(
    mnrg_sum > 1,
    0.5757 * (Pm_outNRG1 + Pm_outNRG2 + Pm_outNRGA) / 2.0,
    0.5757 * (Pm_outNRG1 + Pm_outNRG2 + Pm_outNRGA)
)
# Добавляем дискреты в data (0/1)
for col in ["МНРГа включить", "МНРГ1 включить", "МНРГ2 включить"]:
    if col in discrete_raw:
        data[col] = pd.Series(discrete_raw[col], index=data.index, dtype=float)  # переводим в float для NN
    else:
        data[col] = 0.0  # если нет в raw, заполняем нулями

input_params = ["Pm_outNRG1", "Pm_outNRG2", "Pm_outNRGA", "МНРГа включить", "МНРГ1 включить", "МНРГ2 включить", "N_MNRG1", "N_MNRG2", "N_MNRGa"]
target_param = "Pm_inR"
# model, history, scaler_X, scaler_y = train_simple_nn(data, input_params, target_param, epochs=100)
#
# # --- после обучения модели ---
# # Делаем предсказание на всех данных
# X_all = data[input_params].fillna(0).to_numpy(dtype=float)
# X_all_scaled = scaler_X.transform(X_all)
#
# y_pred_scaled = model.predict(X_all, verbose=0)  # предсказания модели
# y_pred = scaler_y.inverse_transform(y_pred_scaled)  # возвращаем в "реальные" значения
#
# # Добавляем в scaled_data
# scaled_data["Pm_inR_NN"] = y_pred.flatten()

work_nasos(mnrg1, Pm_1, "Pm_outNRG1", "Pm_outNRG1_imit")
work_nasos(mnrg2, Pm_2, "Pm_outNRG2", "Pm_outNRG2_imit")
work_nasos(mnrga, Pm_a, "Pm_outNRGA", "Pm_outNRGa_imit")
# work_nasos(mask_int, Pm_inR, "Pm_inR", "Pm_inR_imit")

# --- Дискреты добавляем в data ---
for col in ["МНРГа включить", "МНРГ1 включить", "МНРГ2 включить"]:
    if col in discrete_raw:
        data[col] = pd.Series(discrete_raw[col], index=data.index, dtype=float)
    else:
        data[col] = 0.0

# Δ = разница между реальным и имитацией
data["Pm_inR_imit"] = pd.Series(analog_raw.get("Pm_inR_imit", np.zeros(len(time_index))), index=data.index, dtype=float)
data["Delta"] = data["Pm_inR"] - data["Pm_inR_imit"]

# обучаем модель на Δ
X = data[input_params].fillna(0).to_numpy()
# y = data["Delta"].fillna(0).to_numpy()
y = data["Pm_inR"].fillna(0).to_numpy()

# # простая нейросеть
# model = keras.Sequential([
#     layers.Input(shape=(X.shape[1],)),
#     layers.Dense(32, activation="elu"),
#     layers.Dropout(0.1),
#     layers.Dense(16, activation="elu"),
#     layers.Dense(1)
# ])
# # model.compile(optimizer="adam", loss="mse")
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#               loss=keras.losses.Huber(), metrics=["mae"])
# model.fit(X, y, epochs=800, batch_size=16, verbose=1)
#
# # предсказание Δ
# y_delta_pred = model.predict(X, verbose=0).flatten()
#
# # цифровой двойник = базовая формула + коррекция нейросетью
# # data["Pm_inR_hybrid"] = data["Pm_inR_imit"] + y_delta_pred
# data["Pm_inR_nn"] = y_delta_pred
#
# # обучаем модель на Δ
# X = data[input_params].fillna(0).to_numpy()
# y = data["Delta"].fillna(0).to_numpy()
#
# # простая нейросеть
# model1 = keras.Sequential([
#     layers.Input(shape=(X.shape[1],)),
#     layers.Dense(32, activation="elu"),
#     layers.Dropout(0.1),
#     layers.Dense(16, activation="elu"),
#     layers.Dense(1)
# ])
# # model.compile(optimizer="adam", loss="mse")
# model1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#               loss=keras.losses.Huber(), metrics=["mae"])
# model1.fit(X, y, epochs=800, batch_size=16, verbose=1)
#
# # предсказание Δ
# y_delta_pred1 = model1.predict(X, verbose=0).flatten()
#
# # цифровой двойник = базовая формула + коррекция нейросетью
# data["Pm_inR_hybrid"] = data["Pm_inR_imit"] + y_delta_pred1
#
# # Добавляем нормализованные значения в scaled_data
# scaled_data["Pm_inR_imit"] = (data["Pm_inR_imit"] - (data["Pm_inR"].min() - (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min()))) / ((data["Pm_inR"].max() + (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min())) - (data["Pm_inR"].min() - (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min())))
# scaled_data["Pm_inR_hybrid"] = (data["Pm_inR_hybrid"] - (data["Pm_inR"].min() - (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min()))) / ((data["Pm_inR"].max() + (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min())) - (data["Pm_inR"].min() - (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min())))
# scaled_data["Pm_inR_nn"] = (data["Pm_inR_nn"] - (data["Pm_inR"].min() - (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min()))) / ((data["Pm_inR"].max() + (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min())) - (data["Pm_inR"].min() - (0.1 * data["Pm_inR"].max() - data["Pm_inR"].min())))



# === Построение интерактивного графика (Plotly) ===
fig = go.Figure()

# for i, p in enumerate(params):
#     if p in scaled_data:
#         # customdata: для аналогов — оригинальные значения, для дискретов — реальные 0/1 из discrete_raw
#         if p in data.columns:
#             custom = pd.to_numeric(data[p], errors='coerce').to_numpy()
#         else:
#             custom = discrete_raw.get(p, (scaled_data[p] - levels_map[p]))  # prefer raw 0/1 if available
#         fig.add_trace(go.Scatter(
#             x=time_index,
#             y=scaled_data[p],
#             mode='lines',
#             name=p,
#             line_shape='hv',  # дискретные — ступенчатые линии
#             customdata=custom,
#             hovertemplate=(f"<b>{p}</b><br>Время: %{{x|%H:%M:%S}}<br>Значение: %{{customdata:.3f}}<extra></extra>")
#         ))
# customdata: для аналогов — оригинальные значения, для дискретов — реальные 0/1 из discrete_raw
custom = pd.to_numeric(data["Pm_inR"], errors='coerce').to_numpy()
fig.add_trace(go.Scatter(
    x=time_index,
    y=scaled_data["Pm_inR"],
    mode='lines',
    name="Oil pressure at the gearbox inlet",
    line_shape='hv',  # дискретные — ступенчатые линии
    customdata=custom,
    hovertemplate=(f"</b><br>Время: %{{x|%H:%M:%S}}<br>Значение (МПа): %{{customdata:.3f}}<extra></extra>")
))

grafic_imit("Pm_outNRG1_imit")
grafic_imit("Pm_outNRG2_imit")
grafic_imit("Pm_outNRGa_imit")
grafic_imit("Pm_inR_imit")

if "Pm_inR" in scaled_data and "Pm_inR_hybrid" in scaled_data:
    fig.add_trace(go.Scatter(
        x=time_index,
        y=scaled_data["Pm_inR_hybrid"],
        mode='lines',
        name="Pm_inR (гибрид)",
        line=dict(color='purple', dash='dashdot'),
        customdata = pd.to_numeric(data["Pm_inR_hybrid"], errors='coerce').to_numpy(),  # сырые данные
        hovertemplate = (
            "<br>" +
            # "Норм.: %{y:.3f}<br>" +  # нормализованное (отображаемое на графике)
            "Значение: %{customdata:.3f}"  # исходное (для подсказки)
        ),

    ))
if "Pm_inR" in scaled_data and "Pm_inR_nn" in scaled_data:
    fig.add_trace(go.Scatter(
        x=time_index,
        y=scaled_data["Pm_inR_nn"],
        mode='lines',
        name="Pm_inR (нейросеть)",
        line=dict(color='green', dash='dot'),
        customdata = pd.to_numeric(data["Pm_inR_nn"], errors='coerce').to_numpy(),  # сырые данные
        hovertemplate = (
            "<br>" +
            # "Норм.: %{y:.3f}<br>" +  # нормализованное (отображаемое на графике)
            "Значение: %{customdata:.3f}"  # исходное (для подсказки)
        ),

    ))
# Макет
fig.update_layout(
    # title="Параметры с отдельными шкалами",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    hovermode="x unified",
    margin=dict(l=80, r=50, t=80, b=50),
    xaxis=dict(title="Time", showgrid=True, tickformat='%H:%M:%S'),
    yaxis=dict(title="Discharge pressure (MPa)", showticklabels=True, showgrid=True)
)

Горизонтальные линии и подписи слева
for p in params:
    if p in scaled_data:
        lvl = levels_map[p]
        fig.add_hline(y=lvl, line=dict(color="black", width=1))
        fig.add_annotation(xref='paper', x=-0.02, y=lvl + 0.5, text=p, showarrow=False, font=dict(size=20), yref='y')

fig.show()
