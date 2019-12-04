import numpy as np
import pandas as pd
import json
import time

# expected runtime 75mins

starttime = time.process_time()  # calculate time


def reduce_mem_usage(df, verbose=True):
    """
    Function to reduce DF size. Citation:
    https://www.kaggle.com/caesarlupum/ds-bowl-start-here-a-gentle-
    introduction#5.-Reducing-Memory-Size
    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min) and  \
                        (c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)

                elif (c_min > np.iinfo(np.int16).min) and \
                        (c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)

                elif (c_min > np.iinfo(np.int32).min) and \
                        (c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)

                elif (c_min > np.iinfo(np.int64).min) and \
                        (c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)

            else:
                if (c_min > np.finfo(np.float16).min) and \
                        (c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)

                elif (c_min > np.finfo(np.float32).min) and \
                        (c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)

                elif (c_min > np.finfo(np.float64).min) and \
                        (c_max < np.finfo(np.float64).max):
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% '
                      'reduction)'.format(
                        end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df


# def clean_train(df, new_df):
#     """
#     clean train.csv
#     keep event_code = 4100 in assesssment except Bird assessment of
#     which event_code = 4110
#     """
#     for index, row in df.iterrows():
#         if row['event_code'] == 4100 and row['title'].startswith(
#                 ('Cart', 'Cauldron', 'Chest', 'Mushroom')):
#             new_df = new_df.append(row)
#         elif row['event_code'] == 4110 and row['title'].startswith('Bird'):
#             new_df = new_df.append(row)
#
#     return new_df


def increase_true_false_cols(df):
    for i in df.index:
        if json.loads(df.loc[i, 'event_data'])['correct'] is True:
            df.loc[i, 'True'] = 1
            df.loc[i, 'False'] = 0
        elif json.loads(df.loc[i, 'event_data'])['correct'] is False:
            df.loc[i, 'True'] = 0
            df.loc[i, 'False'] = 1

    return df


root = "/Users/a117/Desktop/kaggle/data-science-bowl-2019/data/"
train = pd.read_csv(root + "train.csv")
train = reduce_mem_usage(train)

cleaned_train = train[((train.event_code == 4100)
              & (train.title.str.startswith(('Cart', 'Cauldron', 'Chest',
                                             'Mushroom'))))
             | ((train.event_code == 4110) &
                (train.title.str.startswith('Bird')))]

cleaned_train = increase_true_false_cols(cleaned_train)
cleaned_train.rename(columns={'': 'original_index'}, inplace=True)

filename = '/Users/a117/Desktop/kaggle/data-science-bowl-2019/data/' \
           'cleaned_train.csv'
cleaned_train.to_csv(path_or_buf=filename)

endtime = time.process_time()

print('\nRun time:{:.2f}mins'.format((endtime-starttime)/60))
print('clean_data finished')

