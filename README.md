# JME-ESWA

## Introduction

A lot of job openings have been released online, which makes job recommendation more and more important.
Recently, users often enter their preferences into job search websites to receive some job recommendations that they hope to apply for.
To achieve this goal, the following two types of data are available: (1) auxiliary behavior data such as viewing job postings, bookmarking them and (2) explicit preference data such as conditions for a job that each user desires.
Some researchers proposed job recommendation by addressing either of them.
However, they have not focused on simultaneously addressing both (1) and (2) so far. Given this point, we propose a method for job recommendation that employs auxiliary behavior data and each user's explicit preference data simultaneously.
Additionally, our proposed method addresses multiple behavior overlaps and refines the latent representations.
Furthermore, the integration method of the latent representations obtained from each of the two modules considers the consistency of user preferences and the similarity with job postings, enabling a more accurate estimation of user preferences.
Experimental results on our dataset constructed from an actual job search website show that our proposed model outperforms several state-of-the-arts as measured by MRR and nDCG.

## Usage

### Requirements

- [pyenv](https://github.com/pyenv/pyenv)
- [Poetry](https://github.com/python-poetry/poetry)
- You need to install python (>=3.9 and <3.10) via pyenv in advance.

### Setup

```sh
$ poetry env use 3.9.6 # please specify your python version
$ poetry install
```

### Training

```sh
$ poetry run python -m jme.train
```

You can see the usage by the following command.

```sh
$ poetry run python -m jme.train -h
usage: train.py [-h] [--seed SEED] [--dataset [DATASET]] [--behavior_data [BEHAVIOR_DATA]] [--kge [KGE]] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--dim DIM] [--lr LR] [--patience PATIENCE]
                [--Ks [KS]] [--model_path [MODEL_PATH]] [--use_boac USE_BOAC] [--use_bam USE_BAM] [--use_epl USE_EPL] [--use_mbl USE_MBL] [--use_csw USE_CSW]
                [--consistency_weight CONSISTENCY_WEIGHT] [--neg_size NEG_SIZE]

Run JME.

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed.
  --dataset [DATASET]   Choose a dataset from {toy}
  --behavior_data [BEHAVIOR_DATA]
                        Behavior data, the target behavior should be last.
  --kge [KGE]           Choose a KGE method from {trans_e,trans_h,trans_r,dist_mult,compl_ex,kg2e,conv_e}
  --epoch EPOCH         Number of epoch.
  --batch_size BATCH_SIZE
                        Batch size.
  --dim DIM             Embedding size.
  --lr LR               Learning rate.
  --patience PATIENCE   Number of epoch for early stopping.
  --Ks [KS]             Calculate metric@K when evaluating.
  --model_path [MODEL_PATH]
                        Model path for evaluation.
  --use_boac USE_BOAC   0: Without Behavior Overlap Aware Converter, 1: Full model.
  --use_bam USE_BAM     0: Without Behavior Aware Margin Function, 1: Full model.
  --use_epl USE_EPL     0: Without EPL module, 1: Full model.
  --use_mbl USE_MBL     0: Without MBL module, 1: Full model.
  --use_csw USE_CSW     0: Without Consistency and Similarity Weighting, 1: Full model.
  --consistency_weight CONSISTENCY_WEIGHT
                        Consistency weight.
  --neg_size NEG_SIZE   Negative sampling size.
```

### Evaluation

```sh
$ poetry run python -m jme.test --model_path trained_model/toy_dim64_lr0.0001_trans_e/best.pth # please specify your model path
```

## Dataset

Due to privacy and business restrictions, we cannot release our dataset right now.
Instead of our dataset, there is a toy dataset for checking our code functionality.

You can adapt our code for your own dataset with the following dataset format.

### Dataset format

To use our code, the following two types of data are required.

#### Users' Interaction Data

**train_*.txt**

Auxiliary behavior data between users and items.
Please prepare a file for each auxiliary behavior.
The format is below.

```txt
<user_id> <item_id> <item_id> ...
...
```

In our toy dataset, there are `train_view.txt` and `train_fav.txt`.
You can specify another file names via the `behavior_data` argument.

**train.txt**

Target behavior data between users and items.
The format is same as auxiliary behavior data's.

**val.txt**

Target behavior data between users and items for validation.
The format is below.

```txt
<user_id> <item_id>
...
```

**test.txt**

Target behavior data between users and items for evaluation.
The format is same as validation data's.

#### Users' Explicit Preference Data

**kg.txt**

A knowledge graph data representing users' explicit preferences.
The format is below.

```txt
<head_entity_id> <relation_id> <tail_entity_id>
...
```

**user_entity_map.txt**

A mapping data to link user entities in a knowledge graph to users in interaction data.
The format is below.

```txt
<user_entity_id> <user_id>
...
```

**item_entity_map.txt**

A mapping data to link item entities in a knowledge graph to items in interaction data.
The format is below.

```txt
<item_entity_id> <item_id>
...
```

**user_masters.txt**

Explicit preference data of users.
Rows represent users, and columns represent category data.
If user i and category data j are directly linked in the knowledge graph data, specify 1; otherwise, specify 0.

```txt
<0 or 1> <0 or 1> <0 or 1> ...
...
```

**item_masters.txt**

Attribute data of items.
Rows represent items, and columns represent category data.
Specify 1 if item i is directly linked to category data j in the knowledge graph data; otherwise, specify 0.

```txt
<0 or 1> <0 or 1> <0 or 1> ...
...
```

## Citation

If you make use of this code or our algorithm, please cite the following paper.

```txt
TODO
```
