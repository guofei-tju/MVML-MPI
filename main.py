import os
import pandas as pd
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from dgllife.utils import EarlyStopping

from dataset import MolDataSet, collate
from utils import set_random_seed, evaluate
from model import MVP
import config
import warnings

warnings.filterwarnings("ignore")


def train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper):
    for epoch in range(args.epoch):
        model.train()
        one_batch_bar = tqdm(train_loader, ncols=100)
        one_batch_bar.set_description(f'[iter:{args.iter},epoch:{epoch + 1}/{args.epoch}]')
        cur_lr = optimizer.param_groups[0]["lr"]
        for i, batch in enumerate(one_batch_bar):
            batch_smiles, batch_graph, fps_t, labels = batch
            labels = labels.to(args.device)
            batch_graph = batch_graph.to(args.device)
            batch_smiles["smiles"] = batch_smiles["smiles"].to(args.device)
            atom_feats = batch_graph.ndata['h'].to(args.device)
            fps_t = fps_t.to(args.device)
            pred = model(batch_smiles, batch_graph, atom_feats, fps_t)
            acc, precision, recall, f1score, acc_weight = evaluate(labels, pred)
            loss = loss_func(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            one_batch_bar.set_postfix(dict(
                loss=f'{loss.item():.5f}',
                acc=f'{acc * 100:.2f}%'))
        scheduler.step()
        model.eval()
        res = []
        with torch.no_grad():
            for batch in val_loader:
                batch_smiles, batch_graph, fps_t, labels = batch
                labels = labels.to(args.device)
                batch_graph = batch_graph.to(args.device)
                batch_smiles["smiles"] = batch_smiles["smiles"].to(args.device)
                atom_feats = batch_graph.ndata['h'].to(args.device)
                fps_t = fps_t.to(args.device)
                pred = model(batch_smiles, batch_graph, atom_feats, fps_t)
                acc, precision, recall, f1score, _ = evaluate(labels, pred)
                res.append([acc, precision, recall, f1score])
        val_results = pd.DataFrame(res, columns=['acc', 'precision', 'recall', 'f1_score'])
        r = val_results.mean()
        print(
            f"epoch:{epoch}---validation---acc:{r['acc']}---precision:{r['precision']}---recall:{r['recall']}-"
            f"--f1_score:{r['f1_score']}----lr:{cur_lr}")
        early_stop = stopper.step(r['f1_score'], model)
        if early_stop:
            break


def main(args):
    data_path = './data/kegg_dataset.csv'
    dataset = MolDataSet(data_path)

    data_index = []
    file_name = "data_index.txt"
    with open('./data' + "/" + file_name, "r") as f:
        for line in f.readlines():
            line = eval(line)
            data_index.append(line)

    train_dataset = Subset(dataset, data_index[0])
    validate_dataset = Subset(dataset, data_index[1])
    test_dataset = Subset(dataset, data_index[2])
    n_feats = dataset.node_featurizer.feat_size('h')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate)
    val_loader = DataLoader(validate_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)
    mean_results = []
    for iteration in range(args.iterations):
        args.iter = iteration
        model = MVP(num_classes=args.class_num, in_feats=n_feats, hidden_feats=args.hidden_feats,
                    rnn_embed_dim=args.rnn_embed_dim, blstm_dim=args.rnn_hidden_dim, blstm_layers=args.rnn_layers,
                    fp_2_dim=args.fp_dim, dropout=args.p, num_heads=args.head, device=args.device).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        stopper = EarlyStopping(mode='higher', filename=f'{args.output}/net_{iteration}.pkl', patience=15)
        loss_func = torch.nn.BCEWithLogitsLoss()
        train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper)
        stopper.load_checkpoint(model)
        model.eval()
        res = []
        with torch.no_grad():
            for batch in test_loader:
                batch_smiles, batch_graph, fps_t, labels = batch
                labels = labels.to(args.device)
                batch_graph = batch_graph.to(args.device)
                batch_smiles["smiles"] = batch_smiles["smiles"].to(args.device)
                atom_feats = batch_graph.ndata['h'].to(args.device)
                fps_t = fps_t.to(args.device)
                pred = model(batch_smiles, batch_graph, atom_feats, fps_t)
                acc, precision, recall, f1score, _ = evaluate(labels, pred)
                res.append([acc, precision, recall, f1score])

        test_results = pd.DataFrame(res, columns=['acc', 'precision', 'recall', 'f1_score'])
        r = test_results.mean()
        print(f"test_---acc:{r['acc']}---precision:{r['precision']}---recall:{r['recall']}---f1_score:{r['f1_score']}")
        mean_results.append([r['acc'], r['precision'], r['recall'], r['f1_score']])
        test_mean_results = pd.DataFrame(mean_results, columns=['acc', 'precision', 'recall', 'f1_score'])
        r = test_mean_results.mean()
        print(
            f"mean_test_---acc:{r['acc']}---precision:{r['precision']}---recall:{r['recall']}---f1_score:{r['f1_score']}")
        test_mean_results.to_csv(f'{args.output}/10_test_results.csv', index=False)


if __name__ == '__main__':
    args = config.parse()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    main(args)
