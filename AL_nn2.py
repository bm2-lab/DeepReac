import argparse
import dgl
import numpy as np
import pandas as pd
import random
import time
import json
import copy
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from dgllife.utils import CanonicalAtomFeaturizer

from utils import load_dataset,collate_molgraphs,Meter,EarlyStopping
from model import DeepReac

def arg_parse():
    parser = argparse.ArgumentParser(description="DeepReac arguments.")
    parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--outdir", dest="outdir", help="result directory")
    parser.add_argument("--device", dest="device", help="cpu or cuda")
    parser.add_argument("--epochs", dest="num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--batch", dest="batch_size", type=int, help="batch size to train.")
    parser.add_argument("--patience", dest="patience", type=int, help="patience.")
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--decay', dest='weight_decay', type=float, help='Learning rate decay ratio')
    # parser.add_argument("--metric", dest="metric_name", help="rmse or r2")
    parser.add_argument("--suffix", dest="name_suffix", help="suffix added to the output filename")
    parser.add_argument('--pre', dest='pre_ratio', type=float, help='Ratio of dataset for pre-training.')
    parser.add_argument("--select_mode", dest="select_mode", help="method to select data instances")
    parser.add_argument("--select_num", dest="num_selected", type=int, help="Number of data instances to select.")
    parser.add_argument("--sim_num", dest="simulation_num", type=int, help="Number of rounds for simulation.")

    parser.set_defaults(
        ckptdir="ckpt",
        outdir="results",
        dataset="DatasetA",
        device=0,
        lr=0.001,
        weight_decay=0.0,
        batch_size=64,
        num_epochs=100,
        patience=10,
        # metric_name="rmse",
        name_suffix="1",
        pre_ratio=0.1,
        select_mode = "random",
        num_selected = 10,
        simulation_num = 10,
    )
    return parser.parse_args()

def Rank(outfeats, index, predictions=None, outfeats_labeled=None, label=None, select_mode="random", num_selected=10):
    
    if select_mode == "random":
        random.shuffle(index)
        update_list = index[:num_selected]

    elif select_mode == "diversity":
        similarity_list = []
        for i in range(len(outfeats)):
            s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
            s_max = torch.max(s_).item()
            similarity_list.append(s_max)
        df = pd.DataFrame(zip(index,similarity_list),columns=['index','similarity'])
        df_sorted = df.sort_values(by=['similarity'],ascending=True)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    elif select_mode == "adversary":
        label = label.cpu().numpy().reshape(-1)
        predictions = predictions.cpu().numpy().reshape(-1)
        label_list = []
        for i in range(len(outfeats)):
            s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
            idx_max = torch.argmax(s_).item()
            label_list.append(label[idx_max])
            
        diff = abs(predictions - np.array(label_list))
        df = pd.DataFrame(zip(index,diff),columns=['index','diff'])
        df_sorted = df.sort_values(by=['diff'],ascending=False)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    elif select_mode == "greedy":
        pred = predictions.cpu().numpy().reshape(-1)
        df = pd.DataFrame(zip(index,pred),columns=['index','pred'])
        df_sorted = df.sort_values(by=['pred'],ascending=False)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    elif select_mode == "balanced":
        num_selected_ = int(num_selected/2)
        label = label.cpu().numpy().reshape(-1)
        predictions = predictions.cpu().numpy().reshape(-1)
        label_list = []
        for i in range(len(outfeats)):
            s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
            idx_max = torch.argmax(s_).item()
            label_list.append(label[idx_max])
        diff = abs(predictions - np.array(label_list))

        df = pd.DataFrame(zip(index,predictions),columns=['index','pred'])
        df_sorted = df.sort_values(by=['pred'],ascending=False)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected_])

        df2 = pd.DataFrame(zip(index,diff),columns=['index','diff'])
        df2_sorted = df2.sort_values(by=['diff'],ascending=False)
        df2_index = list(df2_sorted['index'].values)
        i = 0
        while len(update_list) != num_selected:
            if df2_index[i] not in update_list:
                update_list.append(df2_index[i])
            i += 1

    # elif select_mode == "balanced":
    #     num_selected_ = int(num_selected/2)
    #     label = label.cpu().numpy().reshape(-1)
    #     predictions = predictions.cpu().numpy().reshape(-1)
    #     label_list = []
    #     for i in range(len(outfeats)):
    #         s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
    #         idx_max = torch.argmax(s_).item()
    #         label_list.append(label[idx_max])
            
    #     diff = abs(predictions - np.array(label_list))
    #     df = pd.DataFrame(zip(index,diff),columns=['index','diff'])
    #     df_sorted = df.sort_values(by=['diff'],ascending=False)
    #     df_index = df_sorted['index'].values
    #     update_list = list(df_index[:num_selected_])

    #     df2 = pd.DataFrame(zip(index,predictions),columns=['index','pred'])
    #     df2_sorted = df2.sort_values(by=['pred'],ascending=False)
    #     df2_index = list(df2_sorted['index'].values)
    #     i = 0
    #     while len(update_list) != num_selected:
    #         if df2_index[i] not in update_list:
    #             update_list.append(df2_index[i])
    #         i += 1

    return update_list

def run_a_train_epoch(epoch, model, data_loader, loss_criterion, optimizer, args, device):
    model.train()
    train_meter = Meter()
    index = []
    outfeat_list = []
    label_list = []
    for batch_id, batch_data in enumerate(data_loader):
        index_, bg, labels, masks, conditions = batch_data
        index += index_
        label_list.append(labels)
        labels, masks = labels.to(device), masks.to(device)
        
        hs = []
        bgs = []
        for bg_ in bg:
            bg_c = copy.deepcopy(bg_)
            h_ = bg_c.ndata.pop('h')
            hs.append(h_)
            bgs.append(bg_c)

        prediction, out_feats = model(bgs,hs,conditions)
        outfeat_list.append(out_feats)
        train_meter.update(prediction, labels, masks)

        if len(index_) > 19:
            loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    # total_score = np.mean(train_meter.compute_metric(args.metric_name))
    outfeats =  torch.cat(outfeat_list)
    label_all = torch.cat(label_list)
    # print('epoch {:d}/{:d}, training {} {:.4f}'.format(
    #     epoch + 1, args.num_epochs, args.metric_name, total_score))

    return outfeats, index, label_all

def run_an_eval_epoch(model, data_loader,args, device):
    model.eval()
    eval_meter = Meter()
    index = []
    outfeat_list = []
    label_list = []
    predict_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            index_, bg, labels, masks, conditions = batch_data
            index += index_
            label_list.append(labels)
            labels, masks = labels.to(device), masks.to(device)
            
            hs = []
            bgs = []
            for bg_ in bg:
                bg_c = copy.deepcopy(bg_)
                h_ = bg_c.ndata.pop('h')
                hs.append(h_)
                bgs.append(bg_c)

            prediction, out_feats = model(bgs,hs,conditions)
            outfeat_list.append(out_feats)
            predict_list.append(prediction)
            eval_meter.update(prediction, labels, masks)
        total_score = []
        for metric in ["rmse", "mae", "r2"]:
            score = np.mean(eval_meter.compute_metric(metric))
            total_score.append(float(score))
        outfeats = torch.cat(outfeat_list)
        predictions = torch.cat(predict_list)
        label_lists = torch.cat(label_list)
    return total_score, outfeats, index, label_lists, predictions

def main():
    args = arg_parse()
    data, c_num = load_dataset(args.dataset)
    loss_fn = nn.MSELoss(reduction='none')
    in_feats_dim = CanonicalAtomFeaturizer().feat_size('h')
    filename = args.dataset+"_GAT_"+args.select_mode #+"_"+str(args.num_selected)

    if args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda:"+str(args.device)

    metrics = []
    steps = []
    candidates = []
    for num in range(args.simulation_num):
        t_start = time.time()
        model = DeepReac(in_feats_dim, len(data[0][1]), c_num, device = device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.to(device)

        first_round = True
        step = []
        metric = []
        candidate = []
        random.shuffle(data)
        labeled = data[:int(args.pre_ratio*len(data))]
        unlabeled = data[int(args.pre_ratio*len(data)):]
        label_ratio = 0.1

        while label_ratio < 0.9:
            if not first_round:
                sample_ = []
                sample_list = []
                for i,sample in enumerate(unlabeled):
                    if sample[0] in update_list:
                        sample_.append(i)
                sample_.sort(reverse=True)
                for i in sample_:
                    sample_list.append(unlabeled.pop(i))
                labeled += sample_list

            train_val_split = [0.8, 0.2]
            train_set, val_set = split_dataset(labeled, frac_list=train_val_split, shuffle=True, random_state=0)
            train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
            val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
            unlabel_loader = DataLoader(dataset=unlabeled, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
            # if args.metric_name == "r2":
            #     stopper = EarlyStopping(mode='higher', patience=args.patience)
            # else:
            #     stopper = EarlyStopping(mode='lower', patience=args.patience)
            stopper = EarlyStopping(mode='lower', patience=args.patience)

            for epoch in range(args.num_epochs):
                out_feat_train, index_train, label_train = run_a_train_epoch(epoch, model, train_loader, loss_fn, optimizer, args, device)
                val_score, out_feat_val, index_val, label_val, predict_val= run_an_eval_epoch(model, val_loader, args, device)
                early_stop = stopper.step(val_score[0], model)
                if early_stop:
                    break
            unlabel_score, out_feat_un, index_un, label_un, predict_un= run_an_eval_epoch(model, unlabel_loader, args, device)
            label_ratio = len(labeled)/len(data)
            # print('round {:d},label_ratio {:.4f}, validation {} {:.4f}, best {} {:.4f}'.format(
            #         num+1, label_ratio, args.metric_name, val_score,
            #         args.metric_name, stopper.best_score))

            metric.append(unlabel_score)
            step.append(label_ratio)

            # outfeat_labeled = torch.cat([out_feat_train,out_feat_val])
            # label_labeled = torch.cat([label_train,label_val])
            update_list = Rank(out_feat_un, index_un, predict_un, out_feat_train,label_train,args.select_mode,args.num_selected)
            first_round = False
            candidate.append(update_list)

        # file_model = filename+"_step"+str(num+1)+"_"+args.name_suffix+".pth"
        # torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.ckptdir,file_model))

        metrics.append(metric)
        steps.append(step)
        candidate_ = []
        for i in range(len(candidate)):
            for j,sample in enumerate(data):
                if sample[0] in candidate[i]:
                    candidate_.append(sample[-1].item())

        candidates.append(candidate_)
        print("The",num+1,"round", (time.time()-t_start)/3600, "小时")

    result = {"scores":metrics, "steps":steps, "candidates":candidates}
    result = json.dumps(result)
    file_metrics = filename+"_rounds"+str(args.simulation_num)+"_"+args.name_suffix+".json"
    with open(os.path.join(args.outdir,file_metrics),'w') as f:
        f.write(result)

if __name__ == "__main__":
    main()
