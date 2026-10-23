import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_preprocess_data(filepath="data/developer_learning_logs.csv"):
    df = pd.read_csv(filepath)
    X = df[['coding_hours', 'tutorial_completion_rate', 'debugging_hours', 'commits_pushed', 'concept_complexity_score']]
    y = df['milestone_achieved']
    return df, X, y

def train_supervised_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return pipeline, metrics

def train_clustering_model(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['coding_hours', 'debugging_hours', 'commits_pushed', 'tutorial_completion_rate']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, cluster_labels