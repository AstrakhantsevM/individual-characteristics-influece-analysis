#!/usr/bin/env python
# coding: utf-8

import itertools

import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import (
roc_curve, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score,
    f1_score, accuracy_score
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

import pyreadstat

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator, FuncFormatter

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import numpy as np

# ### Класс простой модели

# In[ ]:


class simple_model:
    def __init__(self, df, y_cols, X_cols):
        self.df = df
        self.y = df[y_cols]
        self.X = sm.add_constant(df[X_cols])  

        # Обучаем модели
        self.logit = sm.Logit(self.y, self.X).fit()  
        self.probit = sm.Probit(self.y, self.X).fit() 

        # Предсказанные вероятности для логит-модели
        self.pred_logit = self.logit.predict(self.X)
        # Предсказанные вероятности для пробит-модели
        self.pred_probit = self.probit.predict(self.X)

    def show_models(self):
        print('Логит: \n', self.logit.summary())
        print('Пробит: \n', self.probit.summary())

    # Показать ROC кривую
    def show_roc(self, model='logit'):
        if model == 'logit':
            y_pred_proba = self.pred_logit
        elif model == 'probit':
            y_pred_proba = self.pred_probit
        else:
            raise ValueError("Model should be either 'logit' or 'probit'.")

        # ROC-кривая и AUC
        fpr, tpr, thresholds = roc_curve(self.y, y_pred_proba)
        auc_score = roc_auc_score(self.y, y_pred_proba)

        # Построение графика
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def show_cm(self, threshold=0.5, model='logit'):
        if model == 'logit':
            y_pred_label = (self.pred_logit >= threshold).astype(int)
        elif model == 'probit':
            y_pred_label = (self.pred_probit >= threshold).astype(int)
        else:
            raise ValueError("Model should be either 'logit' or 'probit'.")

        # Создаём матрицу путаницы
        cm = confusion_matrix(self.y, y_pred_label)

        # Отображаем матрицу путаницы
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix (Threshold = {threshold})')
        plt.show()

    # Новая функция: визуализация метрик
    def show_metrics(self, threshold=0.5, model='logit'):
        if model == 'logit':
            y_pred_proba = self.pred_logit
        elif model == 'probit':
            y_pred_proba = self.pred_probit
        else:
            raise ValueError("Model should be either 'logit' or 'probit'.")
    
        y_pred_label = (y_pred_proba >= threshold).astype(int)
    
        precision = precision_score(self.y, y_pred_label)
        recall = recall_score(self.y, y_pred_label)
        f1 = f1_score(self.y, y_pred_label)
        accuracy = accuracy_score(self.y, y_pred_label)
    
        metrics = {'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Accuracy': accuracy}
    
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics.keys(), [v * 100 for v in metrics.values()], color="#4CC9F0")
        plt.ylabel('Score (%)')
        plt.ylim(0, 100)
        plt.title(f'Model Metrics (Threshold = {threshold}, Model = {model})')
    
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
    
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


# ### Класс модели на обученной на подвыборке

# In[7]:


class normal_model:
    def __init__(self, df, y_cols, X_cols, test_size=0.3, random_state=100):
        # Разделяем данные на обучающую и тестовую выборки
        self.df = df
        self.y = df[y_cols]
        self.X = sm.add_constant(df[X_cols])
        
        # Разделение на обучающую и тестовую выборки
        # stratify=self.y гарантирует, что классы будут сбалансированы в обучающей и тестовой выборках
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=test_size, 
                                                                                random_state=random_state, 
                                                                                stratify=self.y)
        
        # Обучаем модели на обучающих данных
        self.logit = sm.Logit(self.y_train, self.X_train).fit()  
        self.probit = sm.Probit(self.y_train, self.X_train).fit() 

        # Предсказания для обеих моделей на тестовых данных
        self.pred_logit = self.logit.predict(self.X_test)
        self.pred_probit = self.probit.predict(self.X_test)

        self.optimized_treshold = self.optimize_threshold()

    def show_models(self):
        # Выводим информацию о моделях
        print('Логит: \n', self.logit.summary())
        print('Пробит: \n', self.probit.summary())

    def show_roc(self, model='logit'):
        # Выбираем модель для построения ROC-кривой
        if model == 'logit':
            y_pred_proba = self.pred_logit
        elif model == 'probit':
            y_pred_proba = self.pred_probit
        else:
            raise ValueError("Model should be either 'logit' or 'probit'.")

        # ROC-кривая и AUC
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        # Построение графика
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def show_cm(self, threshold=0.5, model='logit'):
        # Подбираем лучший трешхолд
        threshold = self.optimized_treshold
        
        # Выбираем модель для построения матрицы путаницы
        if model == 'logit':
            y_pred_label = (self.pred_logit >= threshold).astype(int)
        elif model == 'probit':
            y_pred_label = (self.pred_probit >= threshold).astype(int)
        else:
            raise ValueError("Model should be either 'logit' or 'probit'.")

        # Создаем матрицу путаницы
        cm = confusion_matrix(self.y_test, y_pred_label)

        # Отображаем матрицу путаницы
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix (Threshold = {threshold})')
        plt.show()

    def optimize_threshold(self, model='logit'):
        # Выбираем модель для оптимизации порога
        if model == 'logit':
            y_pred_proba = self.pred_logit
        elif model == 'probit':
            y_pred_proba = self.pred_probit
        else:
            raise ValueError("Model should be either 'logit' or 'probit'.")

        # Создание списка порогов
        thresholds = np.arange(0.0, 1.1, 0.01)
        f1_scores = []

        # Перебираем пороги и вычисляем F1-метрику для каждого
        for threshold in thresholds:
            y_pred_label = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(self.y_test, y_pred_label)
            f1_scores.append(f1)

        # Находим лучший порог и F1
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)

        #print(f"Лучший порог: {best_threshold:.2f}, F1: {best_f1:.4f}")
        return best_threshold

    def show_metrics(self, threshold=0.5, model='logit'):
        # Оптимизируем трешхолд, если нужно
        threshold = self.optimized_treshold
    
        if model == 'logit':
            y_pred_proba = self.pred_logit
        elif model == 'probit':
            y_pred_proba = self.pred_probit
        else:
            raise ValueError("Model should be either 'logit' or 'probit'.")
    
        # Предсказанные классы
        y_pred_label = (y_pred_proba >= threshold).astype(int)
    
        # Метрики на ТЕСТОВОЙ выборке
        precision = precision_score(self.y_test, y_pred_label)
        recall = recall_score(self.y_test, y_pred_label)
        f1 = f1_score(self.y_test, y_pred_label)
        accuracy = accuracy_score(self.y_test, y_pred_label)
    
        metrics = {'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Accuracy': accuracy}
    
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics.keys(), [v * 100 for v in metrics.values()], color="#4CC9F0")
        plt.ylabel('Score (%)')
        plt.ylim(0, 100)
        plt.title(f'Model Metrics (Threshold = {threshold:.2f}, Model = {model})')
    
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom')
    
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def show_marginal_effects(self, model='logit'):
        if model != 'logit':
            raise ValueError("Средние предельные эффекты реализованы только для логит-модели.")
    
        # Получаем предельные эффекты
        margeff = self.logit.get_margeff(at='mean', method='dydx')
        summary = margeff.summary_frame()
    
        # Получаем основные значения
        ame = summary['dy/dx']
        std_err = summary['Std. Err.']
    
        # 95% доверительный интервал: AME ± 1.96 * Std_Error
        ci_lower = ame - 1.96 * std_err
        ci_upper = ame + 1.96 * std_err
    
        # Собираем таблицу
        effects_df = pd.DataFrame({
            '95% CI Lower': ci_lower,
            'AME': ame,
            '95% CI Upper': ci_upper
        })
    
        # Округлим для красивого вывода
        effects_df_rounded = effects_df.round(4)
    
        # Выводим таблицу
        print("Средние предельные эффекты (логит-модель):\n")
        print(effects_df_rounded)
    
        # Формируем словарь для возврата
        result_dict = {}
        for var in effects_df.index:
            result_dict[var] = {
                'AME': round(effects_df.at[var, 'AME'], 4),
                '95% CI Lower': round(effects_df.at[var, '95% CI Lower'], 4),
                '95% CI Upper': round(effects_df.at[var, '95% CI Upper'], 4)
            }
    
        return result_dict

    def show_odds_ratios(self):
        if not hasattr(self.logit, 'params'):
            raise ValueError("Логит-модель не обучена или не найдены параметры.")
    
        # Параметры модели (коэффициенты)
        params = self.logit.params
        # Стандартные ошибки
        conf = self.logit.conf_int()
        conf.columns = ['2.5%', '97.5%']
        
        # Вычисляем отношение шансов и доверительный интервал
        odds_ratios = np.exp(params)
        conf_int_low = np.exp(conf['2.5%'])
        conf_int_high = np.exp(conf['97.5%'])
    
        # Собираем таблицу
        odds_df = pd.DataFrame({
            'Odds Ratio': odds_ratios,
            '95% CI Lower': conf_int_low,
            '95% CI Upper': conf_int_high,
        })
    
        # Округлим красиво
        odds_df_rounded = odds_df.round(4)
    
        # Выводим таблицу
        print("Отношение шансов (логит-модель):\n")
        print(odds_df_rounded)
    
        # Возвращаем словарь
        result_dict = {}
        for var in odds_df.index:
            result_dict[var] = {
                'Odds Ratio': round(odds_df.at[var, 'Odds Ratio'], 4),
                '95% CI Lower': round(odds_df.at[var, '95% CI Lower'], 4),
                '95% CI Upper': round(odds_df.at[var, '95% CI Upper'], 4)
            }
    
        return result_dict


# ### Лучший BIC

# In[ ]:


def find_best_bic_model(df, y_col, X_cols, min_features=3):
    y = df[y_col]

    best_bic = float('inf')
    best_model = None
    best_features = None

    max_features = len(X_cols)

    for r in range(min_features, max_features + 1):
        for subset in itertools.combinations(X_cols, r):
            X_subset = df[list(subset)]
            X_subset = sm.add_constant(X_subset)

            try:
                model = sm.Logit(y, X_subset).fit(disp=0)
                bic = model.bic

                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_features = subset

            except Exception as e:
                continue

    print(f'Лучшая комбинация признаков: {best_features}')
    print(f'Лучший BIC: {best_bic:.2f}')
    print(f'Исключены: {set(X_cols) - set(best_features)}')

    return list(best_features)


# ### AME и Odds ratio visualisation functions

# In[ ]:


def plot_with_CI(data_dict, value_key):
    """
    Универсальная функция для построения графиков с доверительными интервалами
    для разных данных (AME или Odds Ratio).

    :param data_dict: Словарь данных, где ключи - года, а значения - данные с ключами признаков.
    :param value_key: Ключ, который указывает на требуемое значение в подсловаре (например, 'AME' или 'Odds Ratio').
    """
    # Список признаков, для которых отображаем значения
    features = list(data_dict['2020'].keys())
    
    # Определяем количество строк и столбцов для subplots (например, по 4 на строку)
    n_cols = 4
    n_rows = (len(features) + n_cols - 1) // n_cols  # Округляем количество строк

    # Создаем фигуру для subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()  # Для удобства работы с axes

    # Годы отсортированы по возрастанию
    years = sorted(map(int, data_dict.keys()))  # Сортируем года с 2015 до 2020

    # Проходим по каждому признаку и создаем график
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Извлекаем значения, нижний и верхний доверительный интервал для каждого года
        value = [data_dict[str(year)][feature][value_key] for year in years]
        lower_CI = [data_dict[str(year)][feature]['95% CI Lower'] for year in years]
        upper_CI = [data_dict[str(year)][feature]['95% CI Upper'] for year in years]

        # Рисуем точку значения и линии доверительных интервалов
        ax.scatter(years, value, color='blue', label=f'{feature} - {value_key}', zorder=5)  # Точки значений
        
        # Линии для доверительных интервалов
        ax.errorbar(years, value, yerr=[np.array(value) - np.array(lower_CI), np.array(upper_CI) - np.array(value)], fmt='o', color='blue', capsize=5, alpha=0.5)

        # Для выделения доверительных интервалов 2020 года:
        if '2020' in map(str, years):
            ax.scatter([2020], [data_dict['2020'][feature][value_key]], color='red', label=f'2020 - {feature} {value_key}', zorder=10)  # Точка для 2020 года
            ax.errorbar([2020], [data_dict['2020'][feature][value_key]], yerr=[np.array([data_dict['2020'][feature][value_key]]) - np.array([data_dict['2020'][feature]['95% CI Lower']]),
                                                                             np.array([data_dict['2020'][feature]['95% CI Upper']]) - np.array([data_dict['2020'][feature][value_key]])], fmt='o', color='red', capsize=5, alpha=0.5, zorder=10)
            
            # Линии по оси X, на уровне нижней и верхней границы доверительного интервала 2020 года
            lower_line = data_dict['2020'][feature]['95% CI Lower']
            upper_line = data_dict['2020'][feature]['95% CI Upper']
            ax.plot([min(years), max(years)], [lower_line, lower_line], color='black', linestyle='--', alpha=0.3, zorder=3)  # Нижняя линия
            ax.plot([min(years), max(years)], [upper_line, upper_line], color='black', linestyle='--', alpha=0.3, zorder=3)  # Верхняя линия

            # Проверка на пересечение линии с доверительными интервалами других лет
            for year in years:
                if year != 2020:
                    # Проверяем пересечение с нижним доверительным интервалом
                    if lower_CI[years.index(year)] <= lower_line <= upper_CI[years.index(year)]:
                        ax.scatter(year, lower_line, color='red', s=25, zorder=6)  # Красная точка на пересечении с нижней линией
                    # Проверяем пересечение с верхним доверительным интервалом
                    if lower_CI[years.index(year)] <= upper_line <= upper_CI[years.index(year)]:
                        ax.scatter(year, upper_line, color='red', s=25, zorder=6)  # Красная точка на пересечении с верхней линией

        # Настройки графика
        ax.set_title(feature)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        ax.grid(True)

    # Настройка пустых подграфиков, если их не хватает
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

