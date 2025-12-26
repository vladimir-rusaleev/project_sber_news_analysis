import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
from sentence_transformers import SentenceTransformer
import torch
import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold
warnings.filterwarnings('ignore')

NEWS_CSV = "final_news_dataset_cleaned_selective.csv"
BERT_MODEL_NAME = "all-MiniLM-L12-v2"
EMB_PCA_COMPONENTS = 16
WINDOW_DAYS = 30
LOOKAHEAD_DAYS = 4
MIN_FALL = -0.02
BATCH_SIZE = 64
RANDOM_STATE = 42
ENSEMBLE_WEIGHTS = [0.4, 0.5, 0.1]
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(path):
    df = pd.read_csv(path)
    df['news_date_dt'] = pd.to_datetime(df['news_date'])
    df = df.sort_values(['ticker', 'news_date_dt']).reset_index(drop=True)
    return df

def compute_embeddings(df, text_col='news_title', model_name=BERT_MODEL_NAME, batch_size=BATCH_SIZE):
    model = SentenceTransformer(model_name, device=device)
    texts = df[text_col].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, model

def find_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def create_sliding_windows(df, embeddings, window_days=WINDOW_DAYS, step_days=1, lookahead_days=LOOKAHEAD_DAYS):
    all_windows = []
    window_id = 0

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('news_date_dt').reset_index(drop=True)
        if ticker_data.empty:
            continue

        first_date = ticker_data['news_date_dt'].min()
        last_date = ticker_data['news_date_dt'].max()
        current_date = first_date

        while current_date <= last_date:
            window_end = current_date + timedelta(days=window_days)

            mask_window = (df['ticker'] == ticker) & (df['news_date_dt'] >= current_date) & (df['news_date_dt'] < window_end)
            window_news = df[mask_window]
            if len(window_news) >= 1:
                lookahead_end = window_end + timedelta(days=lookahead_days)
                mask_future = (df['ticker'] == ticker) & (df['news_date_dt'] >= window_end) & (df['news_date_dt'] < lookahead_end)
                future_news = df[mask_future]

                has_fall = 0
                if len(future_news) > 0:
                    min_return = future_news['return_1d'].min()
                    if min_return < MIN_FALL:
                        has_fall = 1
                else:
                    has_fall = -1

                idxs = window_news.index.values
                emb_window = embeddings[idxs] if len(idxs) > 0 else np.zeros((1, embeddings.shape[1]))
                emb_mean = emb_window.mean(axis=0)
                emb_std = emb_window.std(axis=0)
                emb_max = emb_window.max(axis=0)

                window_features = {
                    'window_id': window_id,
                    'ticker': ticker,
                    'window_start': current_date,
                    'window_end': window_end,
                    'has_fall': has_fall,
                    'news_count': len(window_news),
                    'avg_sentiment': float(window_news['sentiment'].mean()),
                    'min_sentiment': float(window_news['sentiment'].min()),
                    'max_sentiment': float(window_news['sentiment'].max()),
                    'sentiment_std': float(window_news['sentiment'].std()) if window_news['sentiment'].std()==window_news['sentiment'].std() else 0.0,
                    'geo_ratio': float((window_news['news_type'] == 'geopolitical').mean()),
                    'airline_ratio': float((window_news['news_type'] == 'airline').mean()),
                    'china_ratio': float((window_news['geo_region'] == 'China').mean()),
                    'global_ratio': float((window_news['geo_region'] == 'Global').mean()),
                    'middle_east_ratio': float((window_news['geo_region'] == 'Middle East').mean()),
                    'airline_event_ratio': float((window_news['geo_event_type'] == 'airline').mean()),
                    'geo_event_ratio': float((window_news['geo_event_type'] == 'geopolitical').mean()),
                    'positive_ratio': float((window_news['sentiment_class'] == 'positive').mean()),
                    'negative_ratio': float((window_news['sentiment_class'] == 'negative').mean()),
                    'neutral_ratio': float((window_news['sentiment_class'] == 'neutral').mean()),
                    'start_day_of_week': int(current_date.weekday()),
                    'start_month': int(current_date.month),
                    'price_at_start': float(window_news.iloc[0]['price_prev_close']) if len(window_news)>0 else 0.0,
                    'emb_mean': emb_mean.tolist(),
                    'emb_std': emb_std.tolist(),
                    'emb_max': emb_max.tolist(),
                    'avg_title_length': float(window_news['title_length'].mean()) if 'title_length' in window_news.columns else 0,
                    'title_has_exclamation': float(window_news['news_title'].str.contains('!').mean()),
                    'title_has_question': float(window_news['news_title'].str.contains(r'\?').mean()),
                    'unique_regions_count': float(window_news['geo_region'].nunique()),
                    'unique_event_types': float(window_news['geo_event_type'].nunique()),
                    'sentiment_polarity': float((window_news['sentiment_class'] == 'positive').mean() -
                                              (window_news['sentiment_class'] == 'negative').mean()),
                    'sentiment_extremeness': float((window_news['sentiment'] > 0.7).mean() +
                                                  (window_news['sentiment'] < 0.3).mean()),
                    'days_with_news': float(window_news['news_date_dt'].nunique()),
                    'news_density': len(window_news) / window_days,
                }

                all_windows.append(window_features)
                window_id += 1

            current_date += timedelta(days=step_days)

    windows_df = pd.DataFrame(all_windows)
    return windows_df

def agregate_and_apply_pca(windows_df, emb_key_prefix='emb', n_components=EMB_PCA_COMPONENTS, random_state=RANDOM_STATE):
    emb_mean = np.vstack(windows_df['emb_mean'].values)
    emb_std = np.vstack(windows_df['emb_std'].values)
    emb_max = np.vstack(windows_df['emb_max'].values)

    concat = np.hstack([emb_mean, emb_std, emb_max])
    scaler_emb = StandardScaler()
    concat_scaled = scaler_emb.fit_transform(concat)
    pca = PCA(n_components=n_components, random_state=random_state)
    emb_pca = pca.fit_transform(concat_scaled)

    for i in range(emb_pca.shape[1]):
        windows_df[f'emb_pca_{i}'] = emb_pca[:, i]

    return windows_df, pca, scaler_emb

def tune_catboost(X_train_scaled, y_train, random_state=RANDOM_STATE):
    scale_pos_weight = np.sqrt(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
    
    cat = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1, scale_pos_weight],
        random_state=random_state,
        verbose=0,
        bagging_temperature=0.5,
        depth=8,
        iterations=1500,
        l2_leaf_reg=5,
        learning_rate=0.05
    )
    
    cat.fit(X_train_scaled, y_train)
    
    return cat

def tune_lightgbm(X_train_scaled, y_train, random_state=RANDOM_STATE):
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    lgbm = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        colsample_bytree=0.85,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=10,
        n_estimators=1500,
        subsample=0.8
    )
    
    lgbm.fit(X_train_scaled, y_train)
    
    return lgbm

def tune_random_forest(X_train_scaled, y_train, random_state=RANDOM_STATE):
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        bootstrap=False,
        max_depth=8,
        max_features='sqrt',
        min_samples_leaf=3,
        n_estimators=2000
    )
    
    rf.fit(X_train_scaled, y_train)
    
    return rf

def train_ensemble_with_tuning(X_train_scaled, y_train, X_test_scaled, random_state=RANDOM_STATE):
    cat = tune_catboost(X_train_scaled, y_train, random_state)
    y_proba_cat = cat.predict_proba(X_test_scaled)[:, 1]
    lgbm = tune_lightgbm(X_train_scaled, y_train, random_state)
    y_proba_lgbm = lgbm.predict_proba(X_test_scaled)[:, 1]
    rf = tune_random_forest(X_train_scaled, y_train, random_state)
    y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

    # stack = VotingClassifier(
    #     estimators=[('cat', cat), ('xgb', lgbm), ('rf', rf)],
    #     voting='soft', 
    #     weights=None,
    #     n_jobs=-1
    # )

    # stack.fit(X_train_scaled, y_train)
    # y_proba = stack.predict_proba(X_test_scaled)[:, 1]

    y_proba = (
        ENSEMBLE_WEIGHTS[0] * y_proba_cat +
        ENSEMBLE_WEIGHTS[1] * y_proba_lgbm +
        ENSEMBLE_WEIGHTS[2] * y_proba_rf
    )

    return {
        'cat': cat,
        'xgb': lgbm,
        'rf': rf,
        'proba_cat': y_proba_cat,
        'proba_xgb': y_proba_lgbm,
        'proba_rf': y_proba_rf,
        'proba_ensemble': y_proba,
    }



def train_and_eval(windows_df, model_features, pca_model, random_state=RANDOM_STATE):
    ticker_dummies = pd.get_dummies(windows_df['ticker'], prefix='ticker')
    windows_enc = pd.concat([windows_df.reset_index(drop=True), ticker_dummies.reset_index(drop=True)], axis=1)

    ticker_features = [c for c in ticker_dummies.columns]
    full_features = list(model_features) + ticker_features

    X = windows_enc[full_features].fillna(0).astype(float)
    y = windows_enc['has_fall'].astype(int)

    windows_enc = windows_enc.sort_values('window_start').reset_index(drop=True)
    split_idx = int(len(windows_enc) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ensemble_results = train_ensemble_with_tuning(X_train_scaled, y_train, X_test_scaled, random_state)

    y_proba = ensemble_results['proba_ensemble']

    optimal_threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= optimal_threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Без падения', 'С падением'], zero_division=0))

    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC итоговая: {roc_auc:.3f}")

    roc_auc_cat = roc_auc_score(y_test, ensemble_results['proba_cat'])
    roc_auc_xgb = roc_auc_score(y_test, ensemble_results['proba_xgb'])
    roc_auc_rf = roc_auc_score(y_test, ensemble_results['proba_rf'])
    print(f"ROC-AUC CatBoost: {roc_auc_cat:.3f}")
    print(f"ROC-AUC xgb: {roc_auc_xgb:.3f}")
    print(f"ROC-AUC Random Forest: {roc_auc_rf:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nМатрица ошибок:")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    save_model_bundle(
        save_dir="saved_model",
        ensemble_results=ensemble_results,
        scaler=scaler,
        pca_model=pca_model,
        model_features=full_features,
        optimal_threshold=optimal_threshold
    )

def save_model_bundle(
    save_dir,
    ensemble_results,
    scaler,
    pca_model,
    model_features,
    optimal_threshold,
    bert_model_name=BERT_MODEL_NAME
):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(ensemble_results['cat'], os.path.join(save_dir, 'catboost.pkl'))
    joblib.dump(ensemble_results['xgb'], os.path.join(save_dir, 'xgb.pkl'))
    joblib.dump(ensemble_results['rf'], os.path.join(save_dir, 'random_forest.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(pca_model, os.path.join(save_dir, 'pca.pkl'))

    meta = {
        "model_features": model_features,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "optimal_threshold": float(optimal_threshold),
        "bert_model_name": bert_model_name,
        "emb_pca_components": EMB_PCA_COMPONENTS
    }

    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

def main():
    df = load_data(NEWS_CSV)
    embeddings, bert_model = compute_embeddings(df, text_col='news_title')

    windows_df = create_sliding_windows(df, embeddings, window_days=WINDOW_DAYS, step_days=1, lookahead_days=LOOKAHEAD_DAYS)
    windows_df = windows_df[windows_df['has_fall'] > -1]
    windows_df = windows_df.dropna(subset=['has_fall'])
    print(f"окон: {len(windows_df)}, с падениями: {int(windows_df['has_fall'].sum())} ({windows_df['has_fall'].mean()*100:.2f}%)")

    windows_df, pca_model, emb_scaler = agregate_and_apply_pca(windows_df, n_components=EMB_PCA_COMPONENTS)

    base_features = [
        'news_count', 'avg_sentiment', 'min_sentiment', 'max_sentiment', 'sentiment_std',
        'geo_ratio', 'airline_ratio', 'china_ratio', 'global_ratio', 'middle_east_ratio',
        'airline_event_ratio', 'geo_event_ratio', 'positive_ratio', 'negative_ratio', 'neutral_ratio',
        'start_day_of_week', 'start_month', 'price_at_start', 'avg_title_length', 'title_has_exclamation',
        'title_has_question', 'unique_regions_count', 'unique_event_types', 'sentiment_polarity', 'sentiment_extremeness',
        'days_with_news', 'news_density'
    ]
    emb_features = [f'emb_pca_{i}' for i in range(EMB_PCA_COMPONENTS)]
    model_features = base_features + emb_features

    train_and_eval(windows_df, model_features, pca_model)

if __name__ == "__main__":
    main()