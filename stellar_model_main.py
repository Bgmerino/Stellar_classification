import os

import pandas as pd
import argparse
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data_preparation import data_preparation
from src.train_model import model_training
from src.test_model import model_testing


if __name__ == '__main__':
    
    ## Starting the program and getting the input arguments
    parser = argparse.ArgumentParser(description='Run stellar classification model --using_model train,read --mode production,evaluation',
                                     epilog='Please report bugs and issues to Beatriz')

    parser.add_argument('--using_model', metavar='<train,reuse>', type=str, required=True, help='Define if the model needs to be trained or if we are using a saved version')
    parser.add_argument('--mode', metavar='<prediction,evaluation>', type=str, required=True, help='Using the script in prediction or in evaluation mode')

    args = parser.parse_args()

    using_model = args.using_model
    mode = args.mode

    path_=os.path.join( 'configuration', 'conf.yml')
    config = load_config(path_)
    data_path_training = config['data']['train']
    data_path_testing = config['data']['test']
    not_var = config['model']['no_variables']
    saving_model = config['model']['saving_model']

    print('Running the Stellar classification model with the configuration parameters:')
    print('- Data path training: {}'.format(data_path_training))
    print('- Data path testing: {}'.format(data_path_testing))
    print('- Training the model or reusing?: {}'.format(using_model))
    print('- Mode: {}'.format(mode))
    print('- Saving model?: {}'.format(saving_model))

    ## Considering the possibility of having two sets of data: one for training and a second one for prediction/testing
    try:
        df_train=pd.read_csv(data_path_training)
        df_test=pd.read_csv(data_path_testing)
    except:
        print('Error reading the testing data. Lets split the training set in two.' )
        df_ini=pd.read_csv(data_path_training)
        df_train, df_test = train_test_split(df_ini, test_size=0.25, random_state=1234)
    
    # Decide whether a new model needs to be trained with the data provided or to use an already saved one
    if using_model=='train':
        print('------ Training the model ------')
        df_train_prep=data_preparation(df_train,config,training=True,balanced=True)

        feats = [cc for cc in df_train_prep.columns if cc not in not_var]

        model,scaler = model_training(df_train_prep, feats, config)
        date = datetime.today().strftime('%Y%m%d')

        if scaler is not None:
            with open('models/scaler_'+date+'.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        if saving_model=='True':
            alg=config['model']['algorithm']
            with open('models/'+alg+'_'+date+'.pkl', 'wb') as f:
                pickle.dump(model, f)

    elif using_model=='reuse':
        ref_model = config['model']['ref_model']

        print('------ Using the saved model {} ------'.format(ref_model))
        with open('models/'+ref_model+'.pkl', 'rb') as f:
            model = pickle.load(f)

    else:
        raise ValueError('The possible options for using_model parameter are: reuse (to use an existing model) or train')

    # Once the model is ready, let's evaluate it or perform the predictions
    df_test_prep=data_preparation(df_test,config,training=False,balanced=False)
    feats = [cc for cc in df_test_prep.columns if cc not in not_var]

    results=model_testing(df_test_prep,feats,model,mode=mode)

    print('*********************** Model run correctly ***********************')








