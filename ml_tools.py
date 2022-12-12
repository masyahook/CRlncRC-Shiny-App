import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from shiny.types import FileInfo

"""Tools for ML training"""


def feature_select(df):
    '''Feature selection based on tree feature importances
    '''
    from sklearn.ensemble import ExtraTreesClassifier
    y = np.array(df.label)
    X=np.array(df.iloc[:,1:].values)
    
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=100,
                                  random_state=0, n_jobs = -1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    
    best_features = ['label']
    for f in range(X.shape[1]):
        best_features.append(df.columns[indices[f]+1])

    n_best = len(best_features)
    df = df.loc[:,best_features[0:n_best]] # 0 = label

    return df


def get_data(input_data):
    """Get the fata from the data file."""

    # Reading the feature set
    f: list[FileInfo] = input_data.features()
    df = pd.read_csv(f[0]["datapath"], sep=',' if f[0]["datapath"].endswith('.csv') else '\t', index_col=0).fillna(method='pad')

    # Reading negative-label lncRNAs
    f: list[FileInfo] = input_data.neg_target()
    with open(f[0]["datapath"], "r") as f:
        raw_negative_list = list(map(lambda x: x.strip(), f.readlines()))

    # Reading positive-label lncRNAs
    f: list[FileInfo] = input_data.pos_target()
    with open(f[0]["datapath"], "r") as f:
        raw_positive_list = list(map(lambda x: x.strip(), f.readlines()))
    
    # Equalizing the samples from negative and positive sets
    if len(raw_negative_list) < len(raw_positive_list):
        positive_list = list(np.random.choice(raw_positive_list, 150))
        negative_list = raw_negative_list
    else:
        negative_list = list(np.random.choice(raw_negative_list, 150))
        positive_list = raw_positive_list
    
    # Selecting positive and negative samples
    positive_df = df.loc[positive_list,:]
    negative_df = df.loc[negative_list,:]
    positive_df['label'] = 1
    negative_df['label'] = -1
      
    new_df = pd.concat([positive_df, negative_df])

    cols = list(new_df.columns)
    new_cols = cols[:-1]
    new_cols.insert(0,'label')
    new_df = new_df.loc[:,new_cols]
    
    # feature selection
    new_df = feature_select(new_df)
    y = np.array(new_df.label)
    X = new_df.iloc[:,1:]
    
    return X, y


def feature_importance_display(clf, model_id, X):
    if model_id == 'NB':
        importances = pd.DataFrame(
            {
                'positive_odds': clf.feature_log_prob_[0, :],
                'negative_odds': clf.feature_log_prob_[1, :]
            }, 
            index=X.columns
        ).sort_values('positive_odds', ascending=False)
    elif model_id == 'LR':
        importances = pd.Series(clf.coef_[0], index=X.columns).sort_values(ascending=True)
    elif model_id == 'RF':
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
    else:
        raise NotImplementedError(f'The feature importances module is not supporting model_id="{model_id}"..')
    return importances.plot(kind='bar')


def run_ML_pipeline(report, input_data):
    """
    Runs a certain pipeline on the given training data.
    """

    seed = 42
    np.random.seed(seed)

    model_id = input_data.ml_model()
    train_X, train_y = get_data(input_data)

    if model_id == 'NB':
        from sklearn.naive_bayes import BernoulliNB
        clf = BernoulliNB(alpha = 5)
    
    elif model_id == 'kNN':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors = 3)
    
    elif model_id == 'LR':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=10, random_state=seed, n_jobs = -1)

    elif model_id == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, \
                                min_samples_split=2,\
                                random_state=seed, n_jobs = -1)

    elif model_id == 'SVM':
        from sklearn.svm import SVC
        clf = SVC(kernel = 'rbf', probability = True, random_state=seed)

    else:
        raise NotImplementedError(f'The model_id={model_id} is not known!')

    clf.fit(train_X, train_y)
    pred_y = clf.predict(train_X)
    
    if report == 'classification_metrics':
        report_dict =  classification_report(train_y, pred_y, output_dict=True)
        df = pd.DataFrame(report_dict).transpose().reset_index()
        return df

    elif report == 'confusion_matrix':
        conf_mat_plot = ConfusionMatrixDisplay.from_estimator(clf, train_X, train_y).ax_
        return conf_mat_plot

    elif report == 'roc_auc_curve':
        roc_auc_curve_plot = RocCurveDisplay.from_estimator(clf, train_X, train_y).ax_
        return roc_auc_curve_plot

    elif report == 'feature_importance':
        feature_importance_plot = feature_importance_display(clf, model_id, train_X)
        return feature_importance_plot

    else:
        raise NotImplementedError(f'The report={report} is not known!')


def run_PCA(input_data):
    """Run PCA on the given training data"""

    seed = 42
    np.random.seed(seed)

    train_X, train_y = get_data(input_data)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_train_scaled = StandardScaler().fit_transform(train_X)

    pca = PCA().fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)

    f, ax = plt.subplots(figsize=(15, 7))

    scatter = ax.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1], 
        c = train_y,
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    legend1 = ax.legend(*scatter.legend_elements(), title="Label")
    ax.add_artist(legend1)

    return ax


if __name__ == '__main__':
    run_ML_pipeline('classification_metrics', 'NB')