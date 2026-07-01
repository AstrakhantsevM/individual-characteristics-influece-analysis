### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ####
### ИМПОРТ ДАТАСЕТОВ ### ИМПОРТ ДАТАСЕТОВ ### ИМПОРТ ДАТАСЕТОВ ### ИМПОРТ ДАТАСЕТОВ ### ИМПОРТ ДАТАСЕТОВ ### 
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ####

import pyreadstat

import numpy as np
import pandas as pd

import re
import itertools
from pathlib import Path

import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def load_raw_sav_files(folder_path: str) -> dict:
    df_dict = {}
    
    for file_path in Path(folder_path).glob('*.sav'):
        year_match = re.search(r'20\d{2}', file_path.name)
        dict_key = year_match.group(0) if year_match else file_path.stem
        
        try:
            # Читаем стандартно
            temp_df, _ = pyreadstat.read_sav(file_path)
        except Exception:
            # [вывод] Если падает (как в случае с 2018 годом), сразу используем ISO-8859-1
            temp_df, _ = pyreadstat.read_sav(file_path, encoding='ISO-8859-1')
            
        df_dict[dict_key] = temp_df
        
    return df_dict

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### СХЕМА СТОЛБЦОВ ### СХЕМА СТОЛБЦОВ ### СХЕМА СТОЛБЦОВ ### СХЕМА СТОЛБЦОВ ### СХЕМА СТОЛБЦОВ ### СХЕМА СТОЛБЦОВ ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


COLUMN_SCHEMA = {
    'gender': {
        '2020': 'gender',
        '2019': 'gender',
        '2018': 'gender',
        '2017': 'gender',
        '2016': 'gender',
        '2015': 'gender',
    },

    'age': {
        '2020': 'age',
        '2019': 'age',
        '2018': 'age',
        '2017': 'age',
        '2016': 'age',
        '2015': 'age',
    },

    'household_size': {
        '2020': 'hhsize',
        '2019': 'hhsize',
        '2018': 'hhsize',
        '2017': 'hhsize',
        '2016': 'hhsize',
        '2015': 'hhsize',
    },

    'household_income': {
        '2020': 'gemhhinc',
        '2019': 'GEMHHINC',
        '2018': 'GEMHHINC',
        '2017': 'GEMHHINC',
        '2016': 'GEMHHINC',
        '2015': 'GEMHHINC',
    },

    'knows_entrepreneurs': {
        '2020': 'knowenyy',
        '2019': 'KNOWENyy',
        '2018': 'KNOWEN18',
        '2017': 'KNOWENyy',
        '2016': 'KNOWENyy',
        '2015': 'KNOWENyy',
    },

    'sees_opportunity': {
        '2020': 'opportyy',
        '2019': 'OPPORTyy',
        '2018': 'OPPORT18',
        '2017': 'OPPORTyy',
        '2016': 'OPPORTyy',
        '2015': 'OPPORTyy',
    },

    'has_business_skills': {
        '2020': 'suskilyy',
        '2019': 'SUSKILyy',
        '2018': 'SUSKIL18',
        '2017': 'SUSKILyy',
        '2016': 'SUSKILyy',
        '2015': 'SUSKILyy',
    },

    'fears_failure': {
        '2020': 'frfailyy',
        '2019': 'FRFAILyy',
        '2018': 'FRFAIL18',
        '2017': 'FRFAILyy',
        '2016': 'FRFAILyy',
        '2015': 'FRFAILyy',
    },

    'thinks_startup_easy': {
        '2020': 'easystyy',
        '2019': 'EASYSTyy',
        '2018': 'easystart',
        '2017': 'easystart',
        '2016': 'easystart',
        '2015': 'easystart',
    },

    'supports_equality': {
        '2020': 'equaliyy',
        '2019': 'EQUALIyy',
        '2018': 'EQUALI18',
        '2017': 'EQUALIyy',
        '2016': 'EQUALIyy',
        '2015': 'EQUALIyy',
    },

    'entrepreneur_popular': {
        '2020': 'nbgoodyy',
        '2019': 'NBGOODyy',
        '2018': 'NBGOOD18',
        '2017': 'NBGOODyy',
        '2016': 'NBGOODyy',
        '2015': 'NBGOODyy',
    },

    'high_status_view': {
        '2020': 'nbstatyy',
        '2019': 'NBSTATyy',
        '2018': 'NBSTAT18',
        '2017': 'NBSTATyy',
        '2016': 'NBSTATyy',
        '2015': 'NBSTATyy',
    },

    'media_attention': {
        '2020': 'nbmediyy',
        '2019': 'NBMEDIyy',
        '2018': 'NBMEDI18',
        '2017': 'NBMEDIyy',
        '2016': 'NBMEDIyy',
        '2015': 'NBMEDIyy',
    },

    'bstart': {
        '2020': 'bstartyy',
        '2019': 'Bstartyy',
        '2018': 'bstart',
        '2017': 'bstart',
        '2016': 'bstart',
        '2015': 'bstart',
    },

    'bjobst': {
        '2020': 'bjobstyy',
        '2019': 'BJOBSTyy',
        '2018': 'bjobst',
        '2017': 'bjobst',
        '2016': 'bjobst',
        '2015': 'bjobst',
    },

    'country': {
        '2020': 'country',
        '2019': 'country',
        '2018': 'country',
        '2017': 'country',
        '2016': 'country',
        '2015': 'country',
    },
}

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ФИЛЬТРАЦИЯ СТОЛБЦОВ ### ФИЛЬТРАЦИЯ СТОЛБЦОВ ### ФИЛЬТРАЦИЯ СТОЛБЦОВ ### ФИЛЬТРАЦИЯ СТОЛБЦОВ ### ФИЛЬТРАЦИЯ СТОЛБЦОВ ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


def auto_filter_and_rename(raw_dfs: dict) -> dict:
    """
    Отбирает нужные столбцы из исходных датафреймов GEM
    и переименовывает их в единый стандарт.

    Функция не создает новые признаки.
    Функция не чистит пропуски.
    Функция не меняет типы данных.

    Она только делает:
        старые названия столбцов -> новые стандартные названия.
    """

    filtered_dfs = {}

    for year, df in raw_dfs.items():
        year = str(year)

        if year not in next(iter(COLUMN_SCHEMA.values())):
            raise ValueError(
                f"Год {year} не описан в COLUMN_SCHEMA"
            )

        old_to_new = {}

        for new_col, mapping_by_year in COLUMN_SCHEMA.items():
            old_col = mapping_by_year[year]

            if old_col not in df.columns:
                raise KeyError(
                    f"В датафрейме за {year} год нет столбца '{old_col}' "
                    f"для переменной '{new_col}'"
                )

            old_to_new[old_col] = new_col

        selected_cols = list(old_to_new.keys())

        temp_df = df[selected_cols].copy()
        temp_df = temp_df.rename(columns=old_to_new)

        filtered_dfs[year] = temp_df

    return filtered_dfs


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### СОЗДАЕМ НАШИ ФИЧИ ### СОЗДАЕМ НАШИ ФИЧИ ### СОЗДАЕМ НАШИ ФИЧИ ### СОЗДАЕМ НАШИ ФИЧИ ### СОЗДАЕМ НАШИ ФИЧИ ### СОЗДАЕМ НАШИ ФИЧИ ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


def engineer_features(dfs: dict) -> dict:
    """
    Создает признаки так же, как в исходном коде:

    male:
        1 — мужчина
        0 — иначе

    low_income:
        1 — household_income == 33
        0 — иначе

    high_income:
        1 — household_income == 68100
        0 — иначе

    enterpreneur:
        1 — bstart == 1 или bjobst == 1
        0 — иначе

    После этого удаляет:
        bstart, bjobst, household_income, gender
    """

    processed_dfs = {}

    required_cols = [
        'gender',
        'age',
        'household_size',
        'household_income',
        'bstart',
        'bjobst',
    ]

    for year, df in dfs.items():
        temp_df = df.copy()

        missing_cols = [
            col for col in required_cols
            if col not in temp_df.columns
        ]

        if missing_cols:
            raise KeyError(
                f"В датафрейме за {year} год не хватает столбцов: {missing_cols}"
            )

        # Пол.
        # Это ровно логика из твоего старого кода.
        if 2 in temp_df['gender'].unique():
            temp_df['male'] = (temp_df['gender'] == 1).astype(int)
        else:
            temp_df['male'] = (temp_df['gender'] == 'Male').astype(int)

        # Доход.
        # Базовый класс — средний доход.
        temp_df['low_income'] = (
            temp_df['household_income'] == 33
        ).astype(int)

        temp_df['high_income'] = (
            temp_df['household_income'] == 68100
        ).astype(int)

        # Предприниматель.
        # Важно: оставляю название enterpreneur, чтобы совпадало со старым кодом.
        temp_df['enterpreneur'] = (
            (temp_df['bstart'] == 1) |
            (temp_df['bjobst'] == 1)
        ).astype(int)

        # Базовая фильтрация как в старом коде.
        temp_df = temp_df[temp_df['household_size'] > 0]
        temp_df = temp_df[temp_df['age'] > 0]

        # Удаляем исходные столбцы, из которых сделали новые признаки.
        temp_df = temp_df.drop(
            columns=[
                'bstart',
                'bjobst',
                'household_income',
                'gender',
            ]
        )

        processed_dfs[year] = temp_df

    return processed_dfs

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### УДАЛЯЕМ ВЫБРОСЫ ### УДАЛЯЕМ ВЫБРОСЫ ### УДАЛЯЕМ ВЫБРОСЫ ### УДАЛЯЕМ ВЫБРОСЫ ### УДАЛЯЕМ ВЫБРОСЫ ### УДАЛЯЕМ ВЫБРОСЫ ### УДАЛЯЕМ ВЫБРОСЫ ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

def remove_outliers_iqr(dfs: dict, columns: list) -> dict:
    """
    Удаляет выбросы на основе IQR-метода для указанных столбцов.
    """
    cleaned_dfs = {}
    
    for year, df in dfs.items():
        temp = df.copy()
        for col in columns:
            if col in temp.columns:
                Q1 = temp[col].quantile(0.25)
                Q3 = temp[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Оставляем только те строки, где значение в пределах усов
                temp = temp[(temp[col] >= lower_bound) & (temp[col] <= upper_bound)]
                
        cleaned_dfs[year] = temp
        
    return cleaned_dfs

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ПОТЕРИ ПО СТРАНАМ ### ПОТЕРИ ПО СТРАНАМ ### ПОТЕРИ ПО СТРАНАМ ### ПОТЕРИ ПО СТРАНАМ ### ПОТЕРИ ПО СТРАНАМ ### ПОТЕРИ ПО СТРАНАМ ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

def compare_missing_strategies(
    df,
    question_cols,
    country_col='country',
    full_missing_threshold=1.0
):
    """
    Сравнивает стратегии обработки пропусков с учетом возможных
    системных пропусков по странам.

    Возвращает:
    1. summary — компактную таблицу сравнения стратегий;
    2. countries_without_questions — страны, где весь выбранный блок вопросов отсутствует;
    3. missing_diagnostics — диагностическую таблицу пропусков по странам.
    """

    # Доля пропусков по выбранным вопросам внутри каждой страны
    missing_share_by_country = (
        df
        .groupby(country_col)[question_cols]
        .agg(lambda s: s.isna().mean())
    )

    # Вопрос считается полностью отсутствующим, если доля пропусков >= порога
    fully_missing_questions = missing_share_by_country.ge(full_missing_threshold)

    # Страны, где полностью отсутствует весь выбранный блок вопросов
    countries_without_questions = (
        fully_missing_questions
        .loc[fully_missing_questions.all(axis=1)]
        .index
        .tolist()
    )

    # Диагностика по странам
    missing_diagnostics = (
        missing_share_by_country
        .mul(100)
        .round(1)
        .assign(
            fully_missing_questions=fully_missing_questions.sum(axis=1),
            block_fully_missing=fully_missing_questions.all(axis=1)
        )
        .sort_values(
            ['block_fully_missing', 'fully_missing_questions'],
            ascending=False
        )
    )

    n_total = len(df)

    n_dropna = len(df.dropna())

    n_without_problem_countries = len(
        df
        .loc[~df[country_col].isin(countries_without_questions)]
        .dropna()
    )

    n_without_problem_questions = len(
        df
        .drop(columns=question_cols)
        .dropna()
    )

    summary = pd.DataFrame({
        'Стратегия': [
            'Исходные данные',
            'Обычный dropna',
            'Удалить страны без блока вопросов',
            'Удалить проблемные вопросы'
        ],
        'Логика': [
            'До обработки пропусков.',
            'Удаляются все строки с любым пропуском.',
            'Сначала исключаются страны, где весь блок вопросов отсутствует, затем применяется dropna.',
            'Исключаются вопросы с системными пропусками, затем применяется dropna по остальным переменным.'
        ],
        'Наблюдений': [
            n_total,
            n_dropna,
            n_without_problem_countries,
            n_without_problem_questions
        ]
    })

    summary['Сохранено, %'] = (
        summary['Наблюдений'] / n_total * 100
    ).round(1)

    summary['К dropna, N'] = summary['Наблюдений'] - n_dropna

    summary['К dropna, п.п.'] = (
        summary['Сохранено, %']
        - summary.loc[summary['Стратегия'] == 'Обычный dropna', 'Сохранено, %'].iloc[0]
    ).round(1)

    return summary, countries_without_questions, missing_diagnostics


def style_missing_strategy_table(summary):
    """
    Компактное оформление таблицы сравнения стратегий обработки пропусков
    для Jupyter Notebook.
    """

    def format_int(x):
        return f'{int(x):,}'.replace(',', ' ')

    def format_pct(x):
        return f'{x:.1f}%'

    def format_delta_int(x):
        x = int(x)
        if x > 0:
            return f'+{x:,}'.replace(',', ' ')
        if x < 0:
            return f'{x:,}'.replace(',', ' ')
        return '0'

    def format_delta_pp(x):
        if x > 0:
            return f'+{x:.1f}'
        if x < 0:
            return f'{x:.1f}'
        return '0.0'

    def highlight_key_rows(row):
        strategy = row['Стратегия']

        if strategy == 'Обычный dropna':
            return ['background-color: #fff7e6'] * len(row)

        if strategy == 'Удалить проблемные вопросы':
            return ['background-color: #eef8f0'] * len(row)

        return [''] * len(row)

    return (
        summary
        .style
        .hide(axis='index')
        .format({
            'Наблюдений': format_int,
            'Сохранено, %': format_pct,
            'К dropna, N': format_delta_int,
            'К dropna, п.п.': format_delta_pp
        })
        .apply(highlight_key_rows, axis=1)
        .set_caption('Сравнение стратегий обработки пропусков')
        .set_table_styles([
            {
                'selector': 'caption',
                'props': [
                    ('caption-side', 'top'),
                    ('text-align', 'left'),
                    ('font-size', '17px'),
                    ('font-weight', '700'),
                    ('color', '#193564'),
                    ('padding', '0 0 10px 0')
                ]
            },
            {
                'selector': 'table',
                'props': [
                    ('border-collapse', 'collapse'),
                    ('width', '100%'),
                    ('font-family', 'Arial, sans-serif'),
                    ('font-size', '14px')
                ]
            },
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#f3f6fa'),
                    ('color', '#193564'),
                    ('font-weight', '700'),
                    ('text-align', 'left'),
                    ('padding', '10px 12px'),
                    ('border-bottom', '1px solid #d8dee9'),
                    ('white-space', 'nowrap')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '10px 12px'),
                    ('border-bottom', '1px solid #edf0f5'),
                    ('vertical-align', 'top')
                ]
            },
            {
                'selector': 'tbody tr:hover',
                'props': [
                    ('background-color', '#f8fafc')
                ]
            }
        ])
        .set_properties(
            subset=['Стратегия'],
            **{
                'font-weight': '600',
                'color': '#193564',
                'min-width': '210px',
                'max-width': '240px'
            }
        )
        .set_properties(
            subset=['Логика'],
            **{
                'text-align': 'left',
                'white-space': 'normal',
                'min-width': '360px',
                'max-width': '520px'
            }
        )
        .set_properties(
            subset=[
                'Наблюдений',
                'Сохранено, %',
                'К dropna, N',
                'К dropna, п.п.'
            ],
            **{
                'text-align': 'right',
                'white-space': 'nowrap'
            }
        )
    )

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ####
### СОЗДАЕМ МОДЕЛЬ ### СОЗДАЕМ МОДЕЛЬ ### СОЗДАЕМ МОДЕЛЬ ### СОЗДАЕМ МОДЕЛЬ ### СОЗДАЕМ МОДЕЛЬ ### СОЗДАЕМ МОДЕЛЬ ### СОЗДАЕМ МОДЕЛЬ ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ####

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def prepare_logit_data(
    df,
    target_column,
    feature_columns,
    continuous_columns=None,
    add_constant=True
):
    """
    Готовит данные для логит-модели:
    - оставляет только нужные столбцы;
    - удаляет пропуски;
    - стандартизирует указанные количественные переменные;
    - формирует y и X.

    Возвращает:
    model_data, y, X, scaler
    """

    if continuous_columns is None:
        continuous_columns = []

    model_columns = [target_column] + feature_columns

    model_data = df[model_columns].dropna().copy()

    scaler = None

    if len(continuous_columns) > 0:
        scaler = StandardScaler()

        model_data[continuous_columns] = scaler.fit_transform(
            model_data[continuous_columns]
        )

    y = model_data[target_column]
    X = model_data[feature_columns]

    if add_constant:
        X = sm.add_constant(X, has_constant='add')

    return model_data, y, X, scaler


def fit_logit_model(y, X, disp=False):
    """
    Оценивает логит-модель.
    """

    model = sm.Logit(y, X)
    result = model.fit(disp=disp)

    return result


def predict_logit_classes(model_result, X, threshold=0.5):
    """
    Получает предсказанные вероятности и классы.
    """

    predicted_probability = model_result.predict(X)

    predicted_class = (
        predicted_probability >= threshold
    ).astype(int)

    return predicted_probability, predicted_class


def get_classification_metrics(y_true, predicted_class, predicted_probability):
    """
    Возвращает таблицу метрик качества классификации.
    """

    metrics = pd.DataFrame({
        'Метрика': [
            'Precision',
            'Recall',
            'F1 Score',
            'Accuracy',
            'AUC'
        ],
        'Значение': [
            precision_score(y_true, predicted_class),
            recall_score(y_true, predicted_class),
            f1_score(y_true, predicted_class),
            accuracy_score(y_true, predicted_class),
            roc_auc_score(y_true, predicted_probability)
        ]
    })

    return metrics


def get_sample_description(model_data, y, feature_columns):
    """
    Возвращает краткое описание модельной выборки.
    """

    sample_description = pd.DataFrame({
        'Показатель': [
            'Количество наблюдений',
            'Доля единиц в зависимой переменной',
            'Количество факторов'
        ],
        'Значение': [
            len(model_data),
            y.mean(),
            len(feature_columns)
        ]
    })

    return sample_description


def plot_roc_curve(y_true, predicted_probability, title='ROC Curve'):
    """
    Строит ROC-кривую и возвращает AUC.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        predicted_probability
    )

    auc_score = roc_auc_score(
        y_true,
        predicted_probability
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f'ROC Curve (AUC = {auc_score:.4f})'
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        label='Random Classifier'
    )

    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    return auc_score


def plot_confusion_matrix(
    y_true,
    predicted_class,
    threshold=0.5,
    title=None
):
    """
    Строит матрицу ошибок.
    """

    cm = confusion_matrix(
        y_true,
        predicted_class
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm
    ).plot(cmap='Blues')

    if title is None:
        title = f'Confusion Matrix, Threshold = {threshold}'

    plt.title(title)
    plt.show()

    return cm


def plot_metrics(
    metrics_df,
    title='Model Metrics',
    exclude_metrics=None
):
    """
    Строит столбчатый график метрик.
    По умолчанию можно исключить AUC, если он не нужен на одном графике
    с precision/recall/f1/accuracy.
    """

    if exclude_metrics is None:
        exclude_metrics = []

    plot_data = metrics_df[
        ~metrics_df['Метрика'].isin(exclude_metrics)
    ].copy()

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        plot_data['Метрика'],
        plot_data['Значение'] * 100
    )

    plt.ylabel('Score (%)')
    plt.ylim(0, 100)
    plt.title(title)

    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f'{value:.1f}%',
            ha='center',
            va='bottom'
        )

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def get_target_distribution(y):
    """
    Возвращает распределение зависимой переменной:
    абсолютные частоты и доли классов.
    """

    counts = (
        y
        .value_counts(dropna=False)
        .sort_index()
    )

    distribution_df = pd.DataFrame({
        'Класс': counts.index,
        'Наблюдений': counts.values,
        'Доля': counts.values / counts.values.sum()
    })

    return distribution_df

def plot_target_distribution(
    distribution_df,
    title='Распределение зависимой переменной'
):
    """
    Строит столбчатую диаграмму распределения классов.
    """

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        distribution_df['Класс'].astype(str),
        distribution_df['Доля'] * 100
    )

    plt.xlabel('Класс')
    plt.ylabel('Доля наблюдений, %')
    plt.ylim(0, 100)
    plt.title(title)

    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f'{value:.1f}%',
            ha='center',
            va='bottom'
        )

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def find_best_bic_features(
    y,
    X,
    feature_columns,
    min_features=3
):
    """
    Ищет лучший набор факторов по BIC.

    Использует уже подготовленные y и X из prepare_logit_data().
    То есть:
    - пропуски уже удалены;
    - количественные переменные уже стандартизированы;
    - const уже добавлена.
    """

    best_bic = np.inf
    best_model = None
    best_feature_columns = None

    bic_results = []

    has_constant = 'const' in X.columns

    for n_features in range(min_features, len(feature_columns) + 1):
        for feature_subset in itertools.combinations(feature_columns, n_features):
            feature_subset = list(feature_subset)

            if has_constant:
                X_subset = X[['const'] + feature_subset]
            else:
                X_subset = X[feature_subset]

            try:
                model_result = fit_logit_model(
                    y=y,
                    X=X_subset,
                    disp=False
                )

                bic_results.append({
                    'n_features': n_features,
                    'bic': model_result.bic,
                    'aic': model_result.aic,
                    'llf': model_result.llf,
                    'features': feature_subset
                })

                if model_result.bic < best_bic:
                    best_bic = model_result.bic
                    best_model = model_result
                    best_feature_columns = feature_subset

            except Exception:
                continue

    if best_feature_columns is None:
        raise ValueError('Не удалось оценить ни одну модель. Проверь y, X и список факторов.')

    bic_results = (
        pd.DataFrame(bic_results)
        .sort_values('bic')
        .reset_index(drop=True)
    )

    return best_feature_columns, best_model, bic_results


def fit_train_test_logit_model(
    y,
    X,
    test_size=0.3,
    random_state=100,
    stratify=True,
    disp=False
):
    """
    Разбивает выборку на train/test и обучает logit на train.

    Логика повторяет старый normal_model:
    - train_test_split;
    - stratify по y;
    - модель обучается на train;
    - вероятности считаются на test.
    """

    stratify_values = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values
    )

    logit_result = fit_logit_model(
        y=y_train,
        X=X_train,
        disp=disp
    )

    predicted_probability = logit_result.predict(X_test)

    results = {
        'logit_result': logit_result,

        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,

        'predicted_probability': predicted_probability
    }

    return results


def find_best_classification_threshold(
    y_true,
    predicted_probability,
    start=0.0,
    stop=1.0,
    step=0.01
):
    """
    Подбирает classification threshold через максимизацию F1 Score.
    """

    thresholds = np.arange(start, stop + step, step)

    threshold_results = []

    for threshold in thresholds:
        predicted_class = (
            predicted_probability >= threshold
        ).astype(int)

        f1 = f1_score(y_true, predicted_class)

        threshold_results.append({
            'threshold': threshold,
            'f1_score': f1
        })

    threshold_results = pd.DataFrame(threshold_results)

    best_row = threshold_results.loc[
        threshold_results['f1_score'].idxmax()
    ]

    best_threshold = float(best_row['threshold'])
    best_f1_score = float(best_row['f1_score'])

    return best_threshold, best_f1_score, threshold_results

def get_odds_ratios(
    logit_result,
    round_digits=4
):
    """
    Возвращает odds ratios и 95% доверительные интервалы
    для логит-модели.
    """

    params = logit_result.params

    conf_int = logit_result.conf_int()
    conf_int.columns = ['2.5%', '97.5%']

    odds_ratios_df = pd.DataFrame({
        'Odds Ratio': np.exp(params),
        '95% CI Lower': np.exp(conf_int['2.5%']),
        '95% CI Upper': np.exp(conf_int['97.5%'])
    })

    return odds_ratios_df.round(round_digits)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ВИЗУАЛИЗАЦИЯ OR ### ВИЗУАЛИЗАЦИЯ OR ### ВИЗУАЛИЗАЦИЯ OR ### ВИЗУАЛИЗАЦИЯ OR ### ВИЗУАЛИЗАЦИЯ OR ### ВИЗУАЛИЗАЦИЯ OR ### ВИЗУАЛИЗАЦИЯ OR ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

def plot_with_CI(data_dict, value_key):

    # Приводим ключи годов к строкам, чтобы работало и с '2020', и с 2020.
    normalized_dict = {}

    for year, year_data in data_dict.items():
        year_key = str(year)

        if isinstance(year_data, pd.DataFrame):
            normalized_dict[year_key] = year_data.to_dict(orient='index')
        else:
            normalized_dict[year_key] = year_data

    data_dict = normalized_dict

    # Список признаков — как в старой функции, берем из 2020 года.
    features = list(data_dict['2020'].keys())

    # Определяем количество строк и столбцов для subplots.
    n_cols = 4
    n_rows = (len(features) + n_cols - 1) // n_cols

    # Создаем фигуру для subplots.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    # Годы отсортированы по возрастанию.
    years = sorted(map(int, data_dict.keys()))

    # Проходим по каждому признаку и создаем график.
    for i, feature in enumerate(features):
        ax = axes[i]

        # Извлекаем значения, нижний и верхний доверительный интервал для каждого года.
        value = [
            data_dict[str(year)][feature][value_key]
            for year in years
        ]

        lower_CI = [
            data_dict[str(year)][feature]['95% CI Lower']
            for year in years
        ]

        upper_CI = [
            data_dict[str(year)][feature]['95% CI Upper']
            for year in years
        ]

        # Рисуем точки значений.
        ax.scatter(
            years,
            value,
            color='blue',
            label=f'{feature} - {value_key}',
            zorder=5
        )

        # Рисуем доверительные интервалы.
        ax.errorbar(
            years,
            value,
            yerr=[
                np.array(value) - np.array(lower_CI),
                np.array(upper_CI) - np.array(value)
            ],
            fmt='o',
            color='blue',
            capsize=5,
            alpha=0.5
        )

        # Для выделения доверительных интервалов 2020 года.
        if '2020' in map(str, years):
            ax.scatter(
                [2020],
                [data_dict['2020'][feature][value_key]],
                color='red',
                label=f'2020 - {feature} {value_key}',
                zorder=10
            )

            ax.errorbar(
                [2020],
                [data_dict['2020'][feature][value_key]],
                yerr=[
                    np.array([data_dict['2020'][feature][value_key]])
                    - np.array([data_dict['2020'][feature]['95% CI Lower']]),

                    np.array([data_dict['2020'][feature]['95% CI Upper']])
                    - np.array([data_dict['2020'][feature][value_key]])
                ],
                fmt='o',
                color='red',
                capsize=5,
                alpha=0.5,
                zorder=10
            )

            # Линии по оси X на уровне нижней и верхней границы CI 2020.
            lower_line = data_dict['2020'][feature]['95% CI Lower']
            upper_line = data_dict['2020'][feature]['95% CI Upper']

            ax.plot(
                [min(years), max(years)],
                [lower_line, lower_line],
                color='black',
                linestyle='--',
                alpha=0.3,
                zorder=3
            )

            ax.plot(
                [min(years), max(years)],
                [upper_line, upper_line],
                color='black',
                linestyle='--',
                alpha=0.3,
                zorder=3
            )

            # Проверка пересечения линий CI 2020 с доверительными интервалами других лет.
            for year in years:
                if year != 2020:
                    if lower_CI[years.index(year)] <= lower_line <= upper_CI[years.index(year)]:
                        ax.scatter(
                            year,
                            lower_line,
                            color='red',
                            s=25,
                            zorder=6
                        )

                    if lower_CI[years.index(year)] <= upper_line <= upper_CI[years.index(year)]:
                        ax.scatter(
                            year,
                            upper_line,
                            color='red',
                            s=25,
                            zorder=6
                        )

        # Настройки графика — как в старой функции.
        ax.set_title(feature)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        ax.grid(True)

    # Настройка пустых подграфиков, если их не хватает.
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()