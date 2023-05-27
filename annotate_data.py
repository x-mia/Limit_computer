#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Data Annotator
# Importing

import argparse
import pandas as pd
import numpy as np


# In[ ]:


def annotate_data(df, src_lng, tgt_lng):
    df['correctness'] = pd.Series(dtype='str')
    df['note'] = pd.Series(dtype='str')

    for i, row in df.iterrows():
            src_word = row[src_lng]
            print(src_word)
            trg_word = row[tgt_lng]
            print(trg_word)

            correct = input("correct: ")
            df.at[i, 'correctness'] = correct

            note = input("note: ")
            df.at[i, 'note'] = note
    return df


# In[ ]:


def computing_accuracy_without_L(df):
    yes = df[df['correctness'] == "yes"]
    accuracy = len(yes)/len(df)
    print("Accuracy without limit is: ", accuracy)


# In[ ]:


def computing_accuracy_with_L(df, limit):
    df['score'] = df['score'].astype(float)
    df['position'] = df['position'].astype(float)
    count = 0
    yes = 0

    for i,row in df.iterrows():
        score = row['score']
        position = row['position']
        cor = row['correctness']
        l = limit + (position * 0.01)
        if score > l:
            count = count + 1
            if cor == 'yes':
                yes = yes + 1
    accuracy = yes/count
    print("Accuracy with limit is: ", accuracy)


# In[ ]:


def main(src_lng, tgt_lng, df_path, limit, output):
    print("Loading the dataframe.")
    df = pd.read_csv(df_path)
    print("Annotating data...")
    df = annotate_data(df, src_lng, tgt_lng)
    print("Computing results...")
    computing_accuracy_without_L(df)
    computing_accuracy_with_L(df, limit)
    print("Saving the dataframe.")
    df.to_csv(output, index=False)
    ("Done.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating aligned word embeddings")
    parser.add_argument("--src_lng", type=str, help="Code of the source language")
    parser.add_argument("--tgt_lng", type=str, help="Code of the target language")
    parser.add_argument("--df_path", type=str, help="Path to the dataframe")
    parser.add_argument("--limit", type=float, help="Limit for the cosine similarity")
    parser.add_argument("--output", type=str, help="Path to save annotated dataframe")

    args = parser.parse_args()

    main(args.src_lng, args.tgt_lng, args.df_path, args.limit, args.output)

