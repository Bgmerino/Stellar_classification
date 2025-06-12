import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import shap
import pickle
from datetime import datetime



def model_testing(df,feats,model,mode='production'):
    """

    Function to generate and save the predictions of the model or to evaluate its performance

    :param df: pandas dataframe with the features and labels to predict or evaluate on
    :param feats: list of features to use
    :param model: fitted machine learning model
    :param mode: string indicating whether to evaluate or predict
    :return:
    """

    x_test = df[feats]

    if isinstance(model, SVC):
        date = datetime.today().strftime('%Y%m%d')

        with open('models/scaler_'+date+'.pkl', 'rb') as f:
            scaler = pickle.load(f)

        x_test_scaled = scaler.transform(x_test)
        y_pred = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)

    elif isinstance(model, RandomForestClassifier):
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)

    else:
        raise ValueError('Error evaluating the model passed')

    if mode=='production':
        print('Generating the predictions and saving them')

        result_df = pd.DataFrame({
            'id': df['obj_ID'],
            'prediction': y_pred
        })

        result_df.to_csv('predictions/output.csv', index=False)

    elif mode=='evaluation':

        print('------ Evaluating the model ------')
        y_test = df['class']

        matrix = confusion_matrix(y_test, y_pred)
        accur = metrics.accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

        print('auc:{}, accuracy:{}'.format(auc, accur))
        print('Confusion matrix:')
        print(matrix)

        if isinstance(model, RandomForestClassifier):
            print('Generating shap figures and saving them')
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test)

            for i in range(shap_values.shape[2]):
                print('SHAP Summary Plot for Class {}'.format(i))
                shap.summary_plot(shap_values[:, :, i], x_test, plot_type='violin',show=False)
                plt.savefig('evaluation_figures/shap_summary_class{}.png'.format(i), bbox_inches='tight', dpi=300)
                plt.close()

    return


