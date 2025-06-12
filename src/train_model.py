from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def model_training(df,feats,config):
    """
    Function to train the selected model with the provided data

    :param df: pandas dataframe with the features and labels
    :param feats: list of features that are going to be used for training
    :param config: dictionary object containing configuration parameters
    :return: fitted machine learning model
    """

    y_train = df['class']
    x_train = df[feats]

    sel_model=config['model']['algorithm']

    if sel_model == 'SVM':
        scaler = preprocessing.StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        param = config['parameters']['SVM']

        model = SVC(kernel=param['kernel'], decision_function_shape=param['shape'],probability=True)
        model.fit(x_train_scaled, y_train)

    elif sel_model == 'RandomForestClassifier':

        param = config['parameters']['RF']

        model = RandomForestClassifier(n_estimators=param['threes'], max_depth=param['max_depth'], min_samples_leaf=param['min_samples'], random_state=0)
        model.fit(x_train, y_train)

        scaler=None

    else:
        raise ValueError('The model selected is not between the options. The actual possibilities are: SVM or RandomForestClassifier')

    return model,scaler
