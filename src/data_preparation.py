import numpy as np

def data_preparation(df,config,training=True,balanced=False):

    """

    Function to clean and prepare the data before training/testing

    :param df: pandas dataframe with the data
    :param config: dictionary object containing configuration parameters
    :param training: boolean value indicating whether to use training or testing
    :param balanced: boolean value indicating whether to balance the classes of the data or not
    :return: pandas dataframe with the clean data
    """

    not_var = config['model']['no_variables']
    nan_value=config['model']['nan_value']

    df = df.replace(nan_value, np.nan)

    cat = config['model']['categories']
    for elem in enumerate(cat):
        df = df.replace(elem[1], elem[0])

    for cc in df.columns:
        if cc not in not_var:
            df[cc] = df[cc].fillna(df[cc].mean())

    if training:
        if balanced:
            minimum_size = df[['obj_ID', 'class']].groupby('class').count().min().iloc[0]
            df_balance = df.groupby('class').apply(lambda x: x.sample(minimum_size, replace=True)).reset_index(
                drop=True)
        else:
            df_balance = df

        return df_balance
    else:
        return df




