#!/usr/bin/env python

import os
import sys
import subprocess
import pandas as pd

def strip_quotes(in_string):
    in_string = in_string.strip()
    if in_string.startswith('"'):
        in_string = in_string[1:]
    if in_string.endswith('"'):
        in_string = in_string[:-1]
    return in_string

def main():
    # Labels and supporting data
    audioset_labels = pd.read_csv("class_labels_indices.csv")
    qa_true_count = pd.read_csv("qa_true_counts.csv")
    original_labels = audioset_labels.copy()

    def get_display_name_from_mid(mid):
        return original_labels.loc[original_labels['mid'] == mid].iloc[0]['display_name']
    
    def get_true_count_from_mid(mid):
        return qa_true_count.loc[qa_true_count['label_id'] == mid].iloc[0]['num_true']

    
    # Training and Eval data
    # audioset_unbal_train_df = pd.read_csv("data/audioset/unbalanced_train_segments.csv")
    audioset_bal_train = pd.read_csv("balanced_train_segments-edited.csv")#, delimiter=', ', engine='python')
    audioset_eval = pd.read_csv("eval_segments-edited.csv")#, delimiter=', ', engine='python')

    train_cols = list(audioset_bal_train.columns.values)[3:]
    # print(train_cols)
    
    eval_cols = list(audioset_eval.columns.values)[3:]
    # print(eval_cols)
    
    # Create column with all labels
    audioset_bal_train.fillna('', inplace=True)
    audioset_eval.fillna('', inplace=True)
    audioset_bal_train['all_labels'] = audioset_bal_train[train_cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    audioset_eval['all_labels'] = audioset_bal_train[eval_cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    audioset_bal_train['all_labels'] = audioset_bal_train['all_labels'].str.rstrip(',')
    audioset_eval['all_labels'] = audioset_eval['all_labels'].str.rstrip(',')

    # Strip quotes from every column. They are no longer needed
    for col in train_cols:
        audioset_bal_train[col] = audioset_bal_train[col].apply(strip_quotes)
    for col in eval_cols:
        audioset_eval[col] = audioset_eval[col].apply(strip_quotes)


    train_cols = list(audioset_bal_train.columns.values)
    print(train_cols)
    
    eval_cols = list(audioset_eval.columns.values)
    print(train_cols)

    # Rearragne columns
    audioset_bal_train = audioset_bal_train[['# YTID', ' start_seconds', ' end_seconds', 'all_labels', ' positive_labels', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14']]
    audioset_eval = audioset_eval[['# YTID', ' start_seconds', ' end_seconds', 'all_labels', ' positive_labels', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13']]
    
    # Rearragne columns
    audioset_bal_train.rename(columns={" positive_labels": 'Unnamed: 3'}, inplace=True)
    audioset_eval.rename(columns={" positive_labels": 'Unnamed: 3'}, inplace=True)

    index_list = [
    72,
    73,
    74,
    75,
    77,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136 ]


    # Only use animal labels
    audioset_labels = audioset_labels.iloc[index_list]

    print("\n\nData before filtering....")    
    print("Train shape:", audioset_bal_train.shape)
    print(audioset_bal_train.head())
    print("Eval shape:", audioset_eval.shape)
    print(audioset_eval.head())

    label_list = audioset_labels['mid'].tolist()
    print("Labels:")
    print(label_list)

    def contains_animal_label(positive_labels):
        for label in label_list:
            if label in positive_labels:
                return True
        return False
    
    def contains_only_animal_label(positive_labels):
        for label in label_list:
            if label not in positive_labels:
                return False
        return True

    audioset_bal_train['animal_sound'] = audioset_bal_train['all_labels'].apply(contains_animal_label)
    audioset_eval['animal_sound'] = audioset_eval['all_labels'].apply(contains_animal_label)

    audioset_bal_train['only_animal'] = audioset_bal_train['all_labels'].apply(contains_only_animal_label)
    audioset_eval['only_animal'] = audioset_eval['all_labels'].apply(contains_only_animal_label)


    print("\n\nData after filtering....")    
    
    # Fitler all non-animal data
    #audioset_bal_train = audioset_bal_train[audioset_bal_train[' positive_labels'].isin(label_list)]
    audioset_bal_train = audioset_bal_train[audioset_bal_train['animal_sound'] == True]
    print("Train shape:", audioset_bal_train.shape)
    print(audioset_bal_train.head())
    
    #  print("Unbalanced train data:")
    #  audioset_unbal_train[audioset_unbal_train[' positive_labels'].isin(label_list)]
    #  audioset_unbal_train.shape
    
    audioset_eval = audioset_eval[audioset_eval['animal_sound'] == True]
    print("Eval shape:", audioset_eval.shape)
    print(audioset_eval.head())


    # Save filtered data frames
    print("\nSaving new filtered animal-only data frames....")
    print('../data/balanced_train_segments-animals.csv')
    print('../data/eval_segments-animals.csv')
    print('../data/class_labels_indices-animals.csv')
    audioset_bal_train.to_csv('../data/audioset/balanced_train_segments-animals.csv', header=True, index=False)
    audioset_eval.to_csv('../data/audioset/eval_segments-animals.csv', header=True, index=False)
    audioset_labels.to_csv('../data/audioset/class_labels_indices-animals.csv', header=True, index=False)

    return (audioset_bal_train, audioset_eval, audioset_labels)

if __name__ == "__main__":
    return_object = main()
    print(type(return_object))