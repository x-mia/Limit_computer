#!/usr/bin/env python
# coding: utf-8


# # Getting results from the aligned word embeddings
# Importing

import argparse
import io
import numpy as np
from tqdm import tqdm
from itertools import repeat
import pandas as pd


# Loading aligned embeddings

def load_vec(emb_path, nmax):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if nmax != -1:
                if len(word2id) == nmax:
                    break

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


# Getting nearest neigbours

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, limit):
    word2id = {v: k for k, v in src_id2word.items()}
    ids = []
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    sorted_scores = np.sort(scores)[::-1]
    lim_scores = []
    for sort_s in sorted_scores:
        l = limit + (int(np.where(sorted_scores == sort_s)[0][0]) * 0.01)
        if sort_s > l:
            lim_scores.append(sort_s)
    k_best = scores.argsort()[-(len(lim_scores)):][::-1]
    translations = []
    score = []
    for i, idx in enumerate(k_best):
        translations.append(tgt_id2word[idx])
        score.append(scores[idx])
        ids.append(word2id[word])

    return translations, score, ids, lim_scores



def get_tgt(eval_df, src_lng, tgt_lng, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, limit):
    words = []
    trs = []
    position = []
    scors = []
    rel_scors = []
    ratio_scors = []
    indexes = []

    for _, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], position=0, leave=True):
        word = row[src_lng]
        try:
            translated, score, ids, lim_scores = get_nn(word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, limit)
            words.extend(repeat(word, len(lim_scores)))
            for y in ids:
                indexes.append(y)
                if len(ids) != len(lim_scores):
                    indexes.extend(repeat(y, len(lim_scores)))
            for el in translated:
                trs.append(el)
                position.append(translated.index(el))
            for i in score:
                rel = score[0] - i
                ratio = i / score[0]
                rel_scors.append(rel)
                scors.append(i)
                ratio_scors.append(ratio)
        except KeyError:
            trs.extend(repeat("-", len(lim_scores)))
            position.extend(repeat("-", len(lim_scores)))
            scors.extend(repeat("-", len(lim_scores)))
            rel_scors.extend(repeat("-", len(lim_scores)))
            ratio_scors.extend(repeat("-", len(lim_scores)))
            indexes.extend(repeat("-", len(lim_scores)))
    
    return words, trs, position, scors, rel_scors, ratio_scors, indexes



def making_df(src_lng, tgt_lng, words, trs, position, scors, rel_scors, ratio_scors, indexes):
    df = {}
    df[src_lng] = words
    df[tgt_lng] = trs
    df["position"] = position
    df["score"] = scors
    df["rel_score"] = rel_scors
    df["ratio_score"] = ratio_scors
    df['index'] = indexes
    result = pd.DataFrame(df)
    result = result.drop_duplicates()
    result = result.sort_values([src_lng, 'position'])
    result = result.reset_index(drop=True)
    return result



def computing_accuracy(result, eval_df, src_lng, tgt_lng):
    merged_df = pd.merge(result, eval_df, how='left',indicator=True, on=[src_lng, tgt_lng])
    correct = merged_df[merged_df["_merge"] == 'both']
    precision = len(correct)/len(result)
    recall = len(correct)/len(eval_df)
    print("Precision is: ", precision)
    print("Recall is: ", recall)
    return merged_df


def main(src_lng, tgt_lng, src_path, tgt_path, eval_df, limit, nmax, output):
    print("Loading source embeddings.")
    src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
    print("Loading target embeddings.")
    tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)
    print("Loading evaluation dataframe.")
    eval_df = pd.read_csv(eval_df)
    print("Getting scores and translation equivalents.")
    words, trs, position, scors, rel_scors, ratio_scors, indexes = get_tgt(eval_df, src_lng, tgt_lng, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, limit)
    print("Creating dataframe with results.")
    df = making_df(src_lng, tgt_lng, words, trs, position, scors, rel_scors, ratio_scors, indexes)
    print("Computing results...")
    df = computing_accuracy(df, eval_df, src_lng, tgt_lng)
    print("Saving the dataframe.")
    df.to_csv(output, index=False)
    ("Done.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating aligned word embeddings")
    parser.add_argument("--src_lng", type=str, help="Code of the source language")
    parser.add_argument("--tgt_lng", type=str, help="Code of the target language")
    parser.add_argument("--src_path", type=str, help="Path to the source embeddings")
    parser.add_argument("--tgt_path", type=str, help="Path to the target embeddings")
    parser.add_argument("--eval_df", type=str, help="Path to the evaluation dataframe")
    parser.add_argument("--limit", type=float, help="Limit for the lowest cosine similarity score")
    parser.add_argument("--nmax", type=int, help="The number of loaded embeddings, -1 to disable")
    parser.add_argument("--output", type=str, help="Path to save the dataframe")

    args = parser.parse_args()

    main(args.src_lng, args.tgt_lng, args.src_path, args.tgt_path, args.eval_df, args.limit, args.nmax, args.output)

