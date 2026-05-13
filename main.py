import pandas as pd
import seaborn as sns
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN

import warnings
warnings.filterwarnings('ignore', module='sklearn')

def stratified_shuffle(x, y, random_state=None):
    # Устанавливаем random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Преобразуем в списки для удобной работы
    if hasattr(x, 'values'):
        x = x.values.tolist()
    if hasattr(y, 'values'):
        y = y.values.tolist()

    # Получаем уникальные классы
    unique_classes = np.unique(y)

    shuffled_indices = []

    # Для каждого класса
    for class_label in unique_classes:
        # Находим индексы объектов этого класса
        class_indices = [i for i, label in enumerate(y) if label == class_label]
        n_class = len(class_indices)

        # Перемешиваем индексы объектов этого класса
        np.random.shuffle(class_indices)
        shuffled_indices.extend(class_indices)

    # Применяем индексы к x и y
    x_shuffled = [x[i] for i in shuffled_indices]
    y_shuffled = [y[i] for i in shuffled_indices]

    return x_shuffled, y_shuffled

def add_features(df):
    # Расчет vx, vy, omega
    vx = (4 * (df['V3real'] + df['V1real']) * np.cos(np.radians(30))) / 15
    vy = (4 * (df['V3real'] + df['V1real']) * np.sin(np.radians(30)) - 4 * df['V2real']) / 15
    omega = (df['V1real'] + df['V2real'] + df['V3real']) / 6

    # Расчет Ix, Iy, Iphi
    Ix = (2 * (df['I1'] + df['I3']) * np.cos(np.radians(30))) / 3
    Iy = (2 * (df['I1'] + df['I3']) * np.sin(np.radians(30)) - 2 * df['I2']) / 15
    Iphi = (df['I1'] + df['I2'] + df['I3']) / 3
    Isum = Ix + Iy + Iphi

    # Расчет Tx, Ty, Tz
    Tx = vx / Ix
    Ty = vy / Iy
    Tz = omega / Iphi

    # Добавление в DataFrame
    df['vx'] = vx
    df['vy'] = vy
    df['omega'] = omega
    df['Ix'] = Ix
    df['Iy'] = Iy
    df['Iphi'] = Iphi
    df['Isum'] = Isum
    df['Tx'] = Tx
    df['Ty'] = Ty
    df['Tz'] = Tz

    return df.drop(['N1', 'N2', 'N3'], axis=1)

# ============================================================================
# ЗАГРУЗКА И ОБРАБОТКА ОБУЧАЮЩИХ ДАННЫХ (A+B)
# ============================================================================

# Чтение XLSX файла
file_path = "Data_Set_(A+B).xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# По заданию нужно определять поверхность 5, для бинарной классификации поверхность 5 будет обозначаться как "1", а остальные - "0"
df['Type'] = df['Type'].apply(lambda x: 1 if x == 5 else 0)

# Добавление рассчитанных признаков
df = add_features(df)

x = df.drop('Type', axis=1)  # все признаки
y = df['Type']  # целевая переменная
# x, y = stratified_shuffle(x, y, random_state=42)

# ============================================================================
# МАСШТАБИРОВАНИЕ ДАННЫХ
# ============================================================================

# Создаем и обучаем scaler на обучающих данных
scaler = StandardScaler()
sampler = SMOTE(random_state=42)
x_scaled = scaler.fit_transform(x)
x_balanced, y_balanced = sampler.fit_resample(x_scaled, y)

# ============================================================================
# ПЕРЕБОР ГИПЕРПАРАМЕТРОВ С GRIDSEARCHCV
# ============================================================================

# Базовая модель MLP
mlp = MLPClassifier(max_iter=1200, random_state=42, verbose=False)

# Сетка гиперпараметров для перебора
param_grid = {
    'hidden_layer_sizes': [(50, 25), (100, 50), (80, 40), (100, 50, 25), (80, 40, 20)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam'],
    'alpha': [0.01, 0.03, 0.05]
}

# Создаем GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=cv,
    scoring=['accuracy', 'f1'],
    refit='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(x_balanced, y_balanced)
results = pd.DataFrame(grid_search.cv_results_)

# ============================================================================
# ВЫБОР МОДЕЛИ С МАКСИМАЛЬНЫМ F1-SCORE
# ============================================================================

# Находим модель с максимальным F1-score
best_f1_model = results.loc[results['mean_test_f1'].idxmax()]

print(f"\nЛУЧШАЯ МОДЕЛЬ (по максимальному F1-score):")
for param, value in best_f1_model['params'].items():
    print(f"    {param}: {value}")

# ============================================================================
# ГРАФИК: Все комбинации с выделением лучшей модели
# ============================================================================

# Создаем подписи для каждой комбинации параметров
param_labels = []
for i in range(len(results)):
    hidden = str(results.loc[i, 'param_hidden_layer_sizes'])
    activation = results.loc[i, 'param_activation']
    solver = results.loc[i, 'param_solver']
    alpha = results.loc[i, 'param_alpha']
    param_labels.append(f"{hidden}\n{activation}\n{solver}, α={alpha}")

# Сортируем по F1-score
results_sorted = results.sort_values('mean_test_f1', ascending=False)
param_labels_sorted = [param_labels[i] for i in results_sorted.index]

# Находим позицию лучшей модели
best_pos = np.where(results_sorted.index == best_f1_model.name)[0][0]

# График
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

n_combinations = len(results_sorted)
x_pos = np.arange(n_combinations)

# Accuracy
acc_means = results_sorted['mean_test_accuracy'].values
acc_stds = results_sorted['std_test_accuracy'].values

colors = ['lightgray'] * n_combinations
colors[best_pos] = 'gold'

ax1.bar(x_pos, acc_means, yerr=acc_stds, capsize=3,
        color=colors, edgecolor='black', alpha=0.7,
        error_kw={'linewidth': 1.5, 'capthick': 1.5})
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy для всех комбинаций гиперпараметров (масштабированные данные)', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(param_labels_sorted, rotation=90, ha='center', fontsize=7)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3, axis='y')

# F1-score
f1_means = results_sorted['mean_test_f1'].values
f1_stds = results_sorted['std_test_f1'].values

ax2.bar(x_pos, f1_means, yerr=f1_stds, capsize=3,
        color=colors, edgecolor='black', alpha=0.7,
        error_kw={'linewidth': 1.5, 'capthick': 1.5})
ax2.set_ylabel('F1-score', fontsize=12)
ax2.set_title('F1-score для всех комбинаций гиперпараметров (масштабированные данные)', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(param_labels_sorted, rotation=90, ha='center', fontsize=7)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('РЕЗУЛЬТАТЫ GRID SEARCH (StandardScaler)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# КРОСС-ВАЛИДАЦИЯ ЛУЧШЕЙ МОДЕЛИ
# ============================================================================

print("\n" + "="*60)
print("КРОСС-ВАЛИДАЦИЯ ЛУЧШЕЙ МОДЕЛИ (5-fold)")
print("="*60)

# Получаем параметры лучшей модели
best_params = best_f1_model['params']

# Создаем модель с лучшими параметрами
best_mlp = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    activation=best_params['activation'],
    solver=best_params['solver'],
    alpha=best_params['alpha'],
    max_iter=800,
    random_state=42,
    verbose=False
)

# Выполняем кросс-валидацию для получения метрик по фолдам на масштабированных данных
cv_scores_accuracy = cross_val_score(best_mlp, x_balanced, y_balanced, cv=cv, scoring='accuracy')
cv_scores_f1 = cross_val_score(best_mlp, x_balanced, y_balanced, cv=cv, scoring='f1')

print(f"\nAccuracy по фолдам: {np.round(cv_scores_accuracy, 4)}")
print(f"F1-score по фолдам: {np.round(cv_scores_f1, 4)}")
print(f"\nСредняя Accuracy: {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std():.4f})")
print(f"Средний F1-score: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std():.4f})")

# ============================================================================
# ТЕСТИРОВАНИЕ НА ВЫБОРКЕ C
# ============================================================================

# Загрузка тестовых данных
file_path_new = "Data_Set_C.xlsx"
df_new = pd.read_excel(file_path_new, engine='openpyxl')

df_new['Type'] = df_new['Type'].apply(lambda x: 1 if x == 5 else 0)
df_new = add_features(df_new)

x_test = df_new.drop('Type', axis=1)
y_test = df_new['Type']

# Масштабируем тестовые данные (используем тот же scaler!)
x_test_scaled = scaler.transform(x_test)

# Обучение на всех обучающих данных и предсказание
best_mlp.fit(x_scaled, y)
y_test_pred = best_mlp.predict(x_test_scaled)

# Метрики на тесте
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ C")
print("="*60)
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1-score: {test_f1:.4f}")

# Матрица ошибок
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title(f'Матрица ошибок (тестовые данные C)\nAccuracy = {test_accuracy:.4f}, F1 = {test_f1:.4f}', fontsize=12)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ГРАФИК: Сравнение реальных и предсказанных значений на выборке C
plt.figure(figsize=(15, 5))
indices_test = range(len(y_test))
plt.plot(indices_test, y_test, 'o-', label='Реальный Type', markersize=3, alpha=0.7, linewidth=0.8)
plt.plot(indices_test, y_test_pred, 's--', label='Предсказанный Type', markersize=3, alpha=0.7, linewidth=0.8)
plt.xlabel('Номер строки (индекс)', fontsize=12)
plt.ylabel('Type', fontsize=12)
plt.title(f'Сравнение реальных и предсказанных значений (тестовые данные - C_extended)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Сохранение модели и scaler
joblib.dump(best_mlp, 'best_mlp_model.pkl')
joblib.dump(scaler, 'scaler.pkl')