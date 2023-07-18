# Limit_computer

A script for extracting the target words from cross-lingual word embeddings by either limiting the cosine similarity or selecting k nearest neighbours. The script partially uses code from [MUSE](https://github.com/facebookresearch/MUSE) [(Conneau et al., 2017)](https://arxiv.org/pdf/1710.04087.pdf). Part of the repository is the script for annotating the data manually. In [demo.ipynb](https://github.com/x-mia/Limit_computer/blob/main/demo.ipynb) you can visualize the graphs using a sample of manually annotated dataframe. More information in the article [Parallel, or Comparable? That Is the Question](https://nlp.fi.muni.cz/raslan/2022/paper8.pdf).

### Requirements
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Tqdm](https://tqdm.github.io/)
* [Matplotlib](https://matplotlib.org/)

### Evaluating the cross-lingual embedding model
To evaluate aligned embeddings using the cosine similarity score limit, add the --limit flag for enabling the limit and select a threshold for the lowest cosine similarity, otherwise, the script will use KNN search, in this case, select a threshold for K nearest neighbours, examples:
With limit:
```bash
python eval.py --src_lng SRC_LNG --tgt_lng TGT_LNG --src_path SRC_PATH --tgt_path TGT_PATH --eval_df EVAL_DF --limit LIMIT --treshold TRESHOLD --nmax NMAX --output OUTPUT
```
Example:
```bash
python eval.py --src_lng et --tgt_lng sk --src_path vectors-et.txt --tgt_path vectors-sk.txt --eval_df et-sk.csv --limit --treshold 0.6 --nmax 50000 --output df.csv
```

K nearest neighbours:
```bash
python eval.py --src_lng SRC_LNG --tgt_lng TGT_LNG --src_path SRC_PATH --tgt_path TGT_PATH --eval_df EVAL_DF --treshold TRESHOLD --nmax NMAX --output OUTPUT
```
Example:
```bash
python eval.py --src_lng et --tgt_lng sk --src_path vectors-et.txt --tgt_path vectors-sk.txt --eval_df et-sk.csv --treshold 3 --nmax 50000 --output df.csv
```

### Annotating the data
To manually annotate the data, simply run:
```bash
python annotate_data.py --src_lng SRC_LNG --tgt_lng TGT_LNG --df_path DF_PATH --limit LIMIT --output OUTPUT
```
Example:
```bash
python annotate_data.py --src_lng et --tgt_lng sk --df_path et-sk.csv --limit 0.6 --output annotated_df.csv
```

### References
* Please cite [[1]](https://nlp.fi.muni.cz/raslan/2022/paper8.pdf) if you found the resources in this repository useful.

[1] Denisová, M. (2022). [Parallel, or Comparable? That Is the Question: The Comparison of Parallel and Comparable Data-based Methods for Bilingual Lexicon Induction.](https://nlp.fi.muni.cz/raslan/2022/paper8.pdf) In *Proceedings of the Sixteenth Workshop on Recent Advances in Slavonic Natural Languages Processing, RASLAN 2022*, pp. 3-13. Tribun EU. 

```
@inproceedings{denisova2022,
   author = {Denisová, Michaela},
   title = {Parallel, or Comparable? That Is the Question: The Comparison of Parallel and Comparable Data-based Methods for Bilingual Lexicon Induction},
   booktitle = {Proceedings of the Sixteenth Workshop on Recent Advances in Slavonic Natural Languages Processing, RASLAN 2022},
   pages = {3--13},
   publisher = {Tribun EU},
   url = {https://nlp.fi.muni.cz/raslan/2022/paper8.pdf},
   year = {2022}
}
```
### Related work
* [A. Conneau, G. Lample, L. Denoyer, MA. Ranzato, H. Jégou - *Word Translation Without Parallel Data*, 2017](https://arxiv.org/pdf/1710.04087.pdf)
