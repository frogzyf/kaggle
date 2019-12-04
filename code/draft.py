import os
import numpy as np
import pandas as pd
import time
from reduce_mem_usage import reduce_mem_usage


starttime = time.process_time()


pd.set_option('display.max_columns', 100)


root = "/Users/a117/Desktop/kaggle/data-science-bowl-2019/data/"

keep_columns = ['event_id', 'game_session', 'installation_id', 'event_count',
                'event_code', 'title', 'game_time', 'type', 'world']


# 读取数据
train = pd.read_csv(root + "train.csv", usecols=keep_columns)
test = pd.read_csv(root + "test.csv", usecols=keep_columns)
train_labels = pd.read_csv(root + "train_labels.csv")
specs = pd.read_csv(root + "specs.csv")
sample_submission = pd.read_csv(root + "sample_submission.csv")


# 减少内存使用
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train_labels = reduce_mem_usage(train_labels)
specs = reduce_mem_usage(specs)
sample_submission = reduce_mem_usage(sample_submission)


# print(train.head(), '\n')
# print(train_labels.head(), '\n')
# print(specs.head(), '\n')

# print(len(train['event_id'].unique()))

# print(train.select_dtypes('object').apply(pd.Series.nunique, axis=0))

new_df = pd.DataFrame()
for n in range(0, len(train)):
    if train.loc[n, 'event_code'] == 4100 \
            or train.loc[n, 'event_code'] == 4110:
        new_df[n] = train.loc[n]

print(new_df.head())

endtime = time.process_time()

print('\nRun time:{:.2f}s'.format(endtime-starttime))


