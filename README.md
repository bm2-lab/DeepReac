# DeepReac+:  An universal framework for quantitative modeling of chemical reaction based on deep active learning
## introduction
DeepReac+ is an efficient and universal computational framework for the prediction of chemical reaction outcomes and selection of optimized reaction conditions. To accelerate the automation of chemical synthesis, artificial intelligence technologies have been introduced to predict potential reaction outcomes before experimentation. However, two limitations exist for current computational approaches: 
- There is a lack of universality and generalization in the modeling of various sorts of chemical reactions. 
- A large amount of data are still necessary to train the model for high predictive performance.

Therefore, the main contributions of DeepReac+ are as follows: 
1. Under the framework of DeepReac+, DeepReac is designed as an efficient graph-neural-network-based representation learning model for chemical reaction outcome prediction, where the 2D structures of molecules can serve as inputs for feature representation learning and subsequent prediction with a universal and generalized prediction ability. Such a model can handle any reaction performance prediction task, including those for yield and stereoselectivity. In addition, a mechanism-agnostic embedding strategy is applied to represent inorganic components, which further broadens the application scope of DeepReac. 
2. An active learning strategy is proposed to explore the chemical reaction space efficiently. Such a strategy helps to substantially save costs and time in reaction outcome prediction and optimal reaction condition searching by reducing the number of necessary experiments for model training. Unlike the traditional uncertainty-based sampling strategy applied in active learning, two novel sampling strategies are presented based on the representation of the reaction space for reaction outcome prediction, i.e., diversity-based sampling and adversary-based sampling. In addition, two other sampling strategies, i.e., greed-based sampling and balance-based sampling, are proposed for optimal reaction condition searching. 

The two characteristics of DeepReac+ are beneficial to the development of automated chemical synthesis platforms and enable cost reduction and liberate the scientific workforce from repetitive tasks.
## Workflow
DeepReac+ is based on an active learning loop and consists of the following steps: 
1. Use a small amount of data, i.e. 10% of a dataset, to pretrain a model;
2. Select a few unlabeled data points to be annotated according to a sampling strategy;
3. Retrain the model iteratively after every inclusion until some criteria is satisfied, i.e. the predictive performance of the model or the amount of labeled data points.

In the above procedure, two main parts play crucial roles: 
- A deep-learning model DeepReac for chemical reaction representation learning as well as outcome prediction. The input can include organic and inorganic components. The former is encoded into dense vectors via Molecule GAT module while the latter is first mapped into sparse one-hot vectors and then embedded to dense vectors via an embedding layer. These vectors are then represented as a reaction graph, where each node corresponds to a component and different components can interact with each other through edges. The reaction graph is feed into Reaction GAT module to model node interactions. A capsule layer is applied on the output of Reaction GAT module to estimate the reaction performance such as yield and stereoselectivity.
- An active sampler for experimental design. As a core of active learning, the sampling strategy is designed to distinguish more valuable data from other data. And for different goals, the sampling strategy should be customized to meet the specific need:
    * To predict chemical reaction outcomes, diversity-based sampling and adversary-based sampling perform well.The former assumes that diverse data can provide the model with a global view of the reaction space and improve its generalization capability. The intuition of the latter is that seeing “adversarial” experimental data can make the model robust.
    * To identify the optimal reaction conditions, greed-based sampling and balance-based sampling are more suitable.The former means that the sample predicted to be optimal should be annotated first. The latter combines adversary-based sampling and greed-based sampling.

## Installation
DeepReac+ requires:
- Python 3.6+
- Pytorch 1.2.0+
- DGL 0.5.2+
- RDKit 2018.09.3+

Clone the repository:
```
git clone https://github.com/bm2-lab/DeepReac.git
```
We recommend running the code block in Tutorial with Jupyter notebook.
## Tutorial
### Data preprocessing
Since DeepReac+ is a universal framework for quantitative modeling of chemical reaction, its input has the same form regardless of reaction type and prediction task. For organic components, only 2D molecular structure is needed and any molecule file recognized by RDKit, including SMILES, MOL, SDF, etc., is suitable. For inorganic components, all we need is categorical information and they are represented as one-hot encoding vectors. Note that organic components can also be represented as one-hot encoding vectors and feed into the embedding layer depending on whether the detailed information of molecular structure is necessary.

For illustration purpose, we take the Dataset A [1] as example to show how to build a dataset:

```
import pandas as pd
import torch
from Data.DatasetA import main_test
from utils import name2g

# Read raw data and convert reaction components to graphs
plate1 = main_test.plate1
plate2 = main_test.plate2
plate3 = main_test.plate3
unscaled = pd.read_csv('Data/DatasetA/dataset.csv')
raw = unscaled.values
y = raw[:,-1]
path = "Data/DatasetA/sdf/"
reactions = []
names = []
plates = [plate1,plate2,plate3]
for plate in plates:
    for r in range(plate.rows):
        for c in range(plate.cols):
            cond = plate.layout[r][c].conditions
            g1 = name2g(path,cond['additive'])
            g2 = name2g(path,cond['ligand'])
            g3 = name2g(path,cond['aryl_halide'])
            g4 = name2g(path,cond['base'])
            name = [cond['additive'],cond['ligand'],cond['aryl_halide'],cond['base']]
            reaction = [g1,g2,g3,g4]
            reactions.append(reaction)
            names.append(name)

# remove data points whose yield is absent
nan_list = [696, 741, 796, 797, 884]
index_list = []
for i in range(3960):
    if i not in nan_list:
        index_list.append(i)
        
# Build a dataset
data = []
for i in index_list:
    label = torch.tensor([y[i]*0.01])
    data_ = (str(i),reactions[i],names[i],label)
    data.append(data_)
    
print(data[0])
***
('0', [DGLGraph(num_nodes=11, num_edges=24,
         ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
         edata_schemes={}), DGLGraph(num_nodes=34, num_edges=74,
         ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
         edata_schemes={}), DGLGraph(num_nodes=11, num_edges=22,
         ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
         edata_schemes={}), DGLGraph(num_nodes=21, num_edges=40,
         ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
         edata_schemes={})], ['5-phenylisoxazole', 'XPhos', '1-chloro-4-(trifluoromethyl)benzene', 'P2Et'], tensor([0.1066]))
***
    
```
### Hyperparameter optimization
For illustration purpose, we take the Dataset A [1] as example to show how to optimize the hyperparameters of DeepReac model:

```
import json
from tqdm import tqdm
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgllife.utils import CanonicalAtomFeaturizer
from utils import load_dataset,collate_molgraphs,Meter,getarrindices,get_split
from model import DeepReac

def run_a_train_epoch(epoch, model, data_loader, loss_criterion, optimizer, device):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        index, bg, labels, masks, conditions = batch_data
        labels, masks = labels.to(device), masks.to(device)
        hs = []
        for bg_ in bg:
            h_ = bg_.ndata.pop('h')
            hs.append(h_)

        prediction, out_feats = model(bg,hs,conditions)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
    total_score = np.mean(train_meter.compute_metric("rmse"))
    return total_score

def run_an_eval_epoch(model, data_loader, device):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            index, bg, labels, masks, conditions = batch_data
            labels, masks = labels.to(device), masks.to(device)
            hs = []
            for bg_ in bg:
                h_ = bg_.ndata.pop('h')
                hs.append(h_)

            prediction, out_feats = model(bg,hs,conditions)
            eval_meter.update(prediction, labels, masks)
        total_score = []
        for metric in ["rmse", "mae", "r2"]:
            total_score.append(np.mean(eval_meter.compute_metric(metric)))
    return total_score

dataset = "DatasetA"
name = "Data/"+dataset+"_CV.json" # five-fold random splitting for the outer loop of nested CV
with open(name,'r') as f:
    result = json.load(f)
split_n = 0 # corresponds to CV-split 1, and so on
train = result["train"][split_n]
val = result["val"][split_n]

train_,test = get_split(len(train),5) # five-fold random splitting for the inner loop of nested CV
scores = []

data, c_num = load_dataset(dataset)
batch_size =128
num_epochs = 1000
lr = 0.001
weight_decay = 0.0
parameter = {
'hidden_feats': [[32,32],[16,16]],
'num_heads': [[4,4],[2,2]],
'out_dim': [16,32,128],
'dropout_2': [0,0.1,0.5],
}
# Users can adjust the hyperparameter set to be optimized here.
# For example, the above codes correspond to two-layer GAT,
# but users want to see the results of three-layer GAT.
# The corresponding codes:
# parameter = {
# 'hidden_feats': [[32,32,32],[16,16,16]],
# 'num_heads': [[4,4,4],[2,2,2]],
# 'out_dim': [16,32,128],
# 'dropout_2': [0,0.1,0.5],
# }

param_permutation = [{},]
for k,v in parameter.items():
    new_values = len(v)
    current_exp_len = len(param_permutation)
    for _ in range(new_values-1):
        param_permutation.extend(copy.deepcopy(param_permutation[:current_exp_len]))
    for validx in range(len(v)):
        for exp in param_permutation[validx*current_exp_len:(validx+1)*current_exp_len]:
            if k == 'hidden_feats':
                exp['hidden_feats_0'] = v[validx]
                exp['hidden_feats_1'] = v[validx]
            elif k == 'num_heads':
                exp['num_heads_0'] = v[validx]
                exp['num_heads_1'] = v[validx]
            else:
                exp[k] = v[validx]

for i in range(len(train_)): # inner loop
    train_idx = getarrindices(train,train_[i])
    test_idx = getarrindices(train,test[i])
    train_set = getarrindices(data,train_idx)
    test_set = getarrindices(data,test_idx)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_molgraphs)

    loss_fn = nn.MSELoss(reduction='none')
    in_feats_dim = CanonicalAtomFeaturizer().feat_size('h')
    device = "cuda:0" # or "cpu"

    score = []
    for j in range(len(param_permutation)):

        model = DeepReac(in_feats_dim, len(data[0][1]), c_num, device = device, **param_permutation[j])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.to(device)

        for epoch in tqdm(range(num_epochs)):
            train_score = run_a_train_epoch(epoch, model, train_loader, loss_fn, optimizer, device)
        val_score = run_an_eval_epoch(model, test_loader, device)

        score.append([float(val_score[0]),float(val_score[1]),float(val_score[2])])

    scores.append(score)

scores = np.array(scores)
avg = np.mean(scores,axis=0)
idx = np.argmin(avg[:,0]) # rmse as metrics
param_best = param_permutation[idx] # The best hyperparameters on CV-split 1 of Dataset A
```

### Prediction of chemical reaction outcomes
For illustration purpose, we take the Dataset B [2] as example to show how to train a DeepReac model with active learning:

```
import random
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from dgllife.utils import CanonicalAtomFeaturizer

from utils import load_dataset,collate_molgraphs,EarlyStopping,arg_parse,Rank,run_a_train_epoch,run_an_eval_epoch
from model import DeepReac
```

1. Set up parameters and load a dataset

```
args = arg_parse()
if args.device == "cpu":
    device = "cpu"
else:
    device = "cuda:"+str(args.device)

data, c_num = load_dataset("DatasetB")
random.shuffle(data)
labeled = data[:int(args.pre_ratio*len(data))]
unlabeled = data[int(args.pre_ratio*len(data)):]
train_val_split = [0.8, 0.2]
train_set, val_set = split_dataset(labeled, frac_list=train_val_split, shuffle=True, random_state=0)
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
unlabel_loader = DataLoader(dataset=unlabeled, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
```

2. Pre-train a DeepReac model with 10% of training set

```
loss_fn = nn.MSELoss(reduction='none')
in_feats_dim = CanonicalAtomFeaturizer().feat_size('h')
model = DeepReac(in_feats_dim, len(data[0][1]), c_num, device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
stopper = EarlyStopping(mode='lower', patience=args.patience)
model.to(device)
***
DeepReac(
  (gnn): GAT_adj(
    (gnn_layers): ModuleList(
      (0): GATLayer(
        (gat_conv): GATConv(
          (fc): Linear(in_features=74, out_features=128, bias=False)
          (feat_drop): Dropout(p=0.0, inplace=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (leaky_relu): LeakyReLU(negative_slope=0.2)
          (res_fc): Linear(in_features=74, out_features=128, bias=False)
        )
      )
      (1): GATLayer(
        (gat_conv): GATConv(
          (fc): Linear(in_features=128, out_features=128, bias=False)
          (feat_drop): Dropout(p=0.0, inplace=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (leaky_relu): LeakyReLU(negative_slope=0.2)
          (res_fc): Linear(in_features=128, out_features=128, bias=False)
        )
      )
    )
  )
  (readout): WeightedSumAndMax(
    (weight_and_sum): WeightAndSum(
      (atom_weighting): Sequential(
        (0): Linear(in_features=32, out_features=1, bias=True)
        (1): Sigmoid()
      )
    )
  )
  (gnn_layers): ModuleList(
    (0): GATLayer(
      (gat_conv): GATConv(
        (fc): Linear(in_features=64, out_features=128, bias=False)
        (feat_drop): Dropout(p=0, inplace=False)
        (attn_drop): Dropout(p=0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=64, out_features=128, bias=False)
      )
    )
    (1): GATLayer(
      (gat_conv): GATConv(
        (fc): Linear(in_features=128, out_features=128, bias=False)
        (feat_drop): Dropout(p=0, inplace=False)
        (attn_drop): Dropout(p=0, inplace=False)
        (leaky_relu): LeakyReLU(negative_slope=0.2)
        (res_fc): Linear(in_features=128, out_features=128, bias=False)
      )
    )
  )
  (digits): CapsuleLayer()
  (predict): Sequential(
    (0): Dropout(p=0, inplace=False)
    (1): Linear(in_features=64, out_features=1, bias=True)
  )
)
***

for epoch in tqdm(range(args.num_epochs)):
    out_feat_train, index_train, label_train = run_a_train_epoch(epoch, model, train_loader, loss_fn, optimizer, args, device)
    val_score, out_feat_val, index_val, label_val, predict_val= run_an_eval_epoch(model, val_loader, args, device)
    early_stop = stopper.step(val_score[0], model)
    if early_stop:
        break
unlabel_score, out_feat_un, index_un, label_un, predict_un= run_an_eval_epoch(model, unlabel_loader, args, device)
label_ratio = len(labeled)/len(data)

print("Size of labelled dataset:",100*label_ratio,"%")
print("Model performance on unlabelled dataset: RMSE:",unlabel_score[0],";MAE:",unlabel_score[1],";R^2:",unlabel_score[2])
***
Size of labelled dataset: 9.98263888888889 %
Model performance on unlabelled dataset: RMSE: 0.18187356184689038 ;MAE: 0.1355237513780594 ;R^2: 0.4841254137661606
***
```

3. Select 10 candidates according to adversary-based strategy

```
update_list = Rank(out_feat_un, index_un, predict_un, out_feat_train,label_train,"adversary",10)
```

4. Update the training set

```
sample_ = []
sample_list = []
for i,sample in enumerate(unlabeled):
    if sample[0] in update_list:
        sample_.append(i)
sample_.sort(reverse=True)
for i in sample_:
    sample_list.append(unlabeled.pop(i))
labeled += sample_list
train_set, val_set = split_dataset(labeled, frac_list=train_val_split, shuffle=True, random_state=0)
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
unlabel_loader = DataLoader(dataset=unlabeled, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
```
5. Retrain the DeepReac model until model performance meets your requirement
```
for epoch in tqdm(range(args.num_epochs)):
    out_feat_train, index_train, label_train = run_a_train_epoch(epoch, model, train_loader, loss_fn, optimizer, args, device)
    val_score, out_feat_val, index_val, label_val, predict_val= run_an_eval_epoch(model, val_loader, args, device)
    early_stop = stopper.step(val_score[0], model)
    if early_stop:
        break
unlabel_score, out_feat_un, index_un, label_un, predict_un= run_an_eval_epoch(model, unlabel_loader, args, device)
label_ratio = len(labeled)/len(data)

print("Size of labelled dataset:",100*label_ratio,"%")
print("Model performance on unlabelled dataset: RMSE:",unlabel_score[0],";MAE:",unlabel_score[1],";R^2:",unlabel_score[2])
***
Size of labelled dataset: 10.199652777777777 %
Model performance on unlabelled dataset: RMSE: 0.17963430405055877 ;MAE: 0.13125237822532654 ;R^2: 0.49230837688682616
***
```

### Identification of optimal reaction condition
For illustration purpose, we take the Dataset B [2] as example to show how to identify optimal reaction condition with active learning:

Perform most of the same steps as those listed above except sampling strategy used in Step 3 and stopping criterion used in Step 5.

3. Select 10 candidates according to balance-based strategy

```
update_list = Rank(out_feat_un, index_un, predict_un, out_feat_train,label_train,"balanced",10)

print("Recommended reaction conditions:")
for reaction in data:
    if reaction[0] in update_list:
        print(reaction[2])
***
Recommended reaction conditions:
['1b', '2b', 'CataCXiumA', 'THF', 'NaOH']
['1b', '2b', 'XPhos', 'DMF', 'NaOH']
['1d', '2c', 'P(Cy)3', 'MeOH', 'KOH']
['1d', '2a', 'SPhos', 'MeCN', 'LiOtBu']
['1a', '2a', 'P(Cy)3', 'DMF', 'Et3N']
['1c', '2b', 'CataCXiumA', 'MeCN', 'CsF']
['1b', '2a', 'P(Ph)3', 'DMF', 'CsF']
['1d', '2b', 'CataCXiumA', 'THF', 'NaOH']
['1b', '2a', 'P(Ph)3', 'MeCN', 'CsF']
['1d', '2b', 'AmPhos', 'THF', 'NaOH']
***
# The first two items, i.e. '1b' and '2b', represent the substrates in Dataset B.
# It should be noted that the same product can be obtained by different substrates which only differ in the leaving group. 
# Hence, substrates here are also defined as reaction conditions in a broad sense and they can be optimized.
```
5. Retrain the DeepReac model until optimal reaction condition is found

### Pre-trained DeepReac models
For the convenience of users, we provide pre-trained models for three datasets in `models`:

Dataset | DeepReac model | Reference
---|---|---
A | DatasetA_DeepReac.pth | [1]
B | DatasetB_DeepReac.pth | [2]
C | DatasetC_DeepReac.pth | [3]

They can be loaded with the following codes:
```
ckpt = torch.load("models/DatasetA_DeepReac.pth",map_location="cuda")
model.load_state_dict(ckpt["model_state_dict"])
```

## Contacts
gykxyy@126.com or qiliu@tongji.edu.cn

## References
[1] Ahneman, D. T.;  Estrada, J. G.;  Lin, S.;  Dreher, S. D.; Doyle, A. G., Predicting reaction performance in C-N cross-coupling using machine learning. Science 2018, 360 (6385), 186-190.

[2] Perera, D.;  Tucker, J. W.;  Brahmbhatt, S.;  Helal, C. J.;  Chong, A.;  Farrell, W.;  Richardson, P.; Sach, N. W., A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. Science 2018, 359 (6374), 429-434.

[3] Zahrt, A. F.;  Henle, J. J.;  Rose, B. T.;  Wang, Y.;  Darrow, W. T.; Denmark, S. E., Prediction of higher-selectivity catalysts by computer-driven workflow and machine learning. Science 2019, 363 (6424), eaau5631.

