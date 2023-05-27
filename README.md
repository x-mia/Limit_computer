# Limit_computer

A script for extracting the target words from cross-lingual word embeddings by limiting the cosine similarity. The script partially uses code from [MUSE](https://github.com/facebookresearch/MUSE) [(Conneau et al., 2017)](https://arxiv.org/pdf/1710.04087.pdf). Part of the repository is the script for annotating the data manually. In [demo.ipynb](https://github.com/x-mia/Limit_computer/blob/main/demo.ipynb) you can visualize the graphs using sample of manually annotated dataframe. More information in the article [Parallel, or Comparable? That Is the Question](https://nlp.fi.muni.cz/raslan/2022/paper8.pdf).

### Requirements
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Tqdm](https://tqdm.github.io/)
* [Matplotlib](https://matplotlib.org/)

### Evaluating without limit (using K nearest neighbours)
To evaluate aligned embeddings using K nearest neighbours, simply run:
```bash
python eval.py --src_lng SRC_LNG --tgt_lng TGT_LNG --src_path SRC_PATH --tgt_path TGT_PATH --eval_df EVAL_DF --k_num K_NUM --nmax NMAX --output OUTPUT
```
Example:
```bash
python eval.py --src_lng et --tgt_lng sk --src_path vectors-et.txt --tgt_path vectors-sk.txt --eval_df et-sk.csv --k_num 3 --nmax 50000 --output df.csv
```

### Evaluating with limiting the scores
To evaluate aligned embeddings using the cosine similarity score limit, simply run:
```bash
python eval.py --src_lng SRC_LNG --tgt_lng TGT_LNG --src_path SRC_PATH --tgt_path TGT_PATH --eval_df EVAL_DF --limit LIMIT --nmax NMAX --output OUTPUT
```
Example:
```bash
python eval.py --src_lng et --tgt_lng sk --src_path vectors-et.txt --tgt_path vectors-sk.txt --eval_df et-sk.csv --limit 0.6 --nmax 50000 --output df.csv
```

### Annotating the data
To manually annotate the data, simply run:
```bash
python eval.py --src_lng SRC_LNG --tgt_lng TGT_LNG --df_path DF_PATH --limit LIMIT --output OUTPUT
```
Example:
```bash
python eval.py --src_lng et --tgt_lng sk --df_path et-sk.csv --limit 0.6 --output annotated_df.csv
```

### References
* Please cite [[1]](https://nlp.fi.muni.cz/raslan/2022/paper8.pdf) if you found the resources in this repository useful.

[1] Denisová, M. (2022). [Parallel, or Comparable? That Is the Question: The Comparison of Parallel and Comparable Data-based Methods for Bilingual Lexicon Induction.](https://nlp.fi.muni.cz/raslan/2022/paper8.pdf) Proceedings of the Sixteenth Workshop on Recent Advances in Slavonic Natural Languages Processing, RASLAN 2022, Tribun EU, pp. 4-13. 

```
@inproceedings{denisova2022,
   author = {Denisová, Michaela},
   title = {Parallel, or Comparable? That Is the Question: The Comparison of Parallel and Comparable Data-based Methods for Bilingual Lexicon Induction},
   booktitle = {Proceedings of the Sixteenth Workshop on Recent Advances in Slavonic Natural Languages Processing, RASLAN 2022},
   pages = {4--13},
   publisher = {Tribun EU},
   url = {https://nlp.fi.muni.cz/raslan/2022/paper8.pdf},
   year = {2022}
}
```
### Related work
* [A. Conneau, G. Lample, L. Denoyer, MA. Ranzato, H. Jégou - *Word Translation Without Parallel Data*, 2017](https://arxiv.org/pdf/1710.04087.pdf)
