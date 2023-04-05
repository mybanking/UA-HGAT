from __future__ import division
from __future__ import print_function
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import argparse
import numpy as np
import pickle as pkl
from copy import deepcopy

import matplotlib

matplotlib.use('AGG')  # 或者PDF, SVG或PS

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from generatelabel.pseudo_labeling_util import pseudo_labeling
from generatelabel.utils import save_checkpoint
from utils import load_data, accuracy, dense_tensor_to_sparse, resample, makedirs
import warnings

warnings.filterwarnings("ignore")
from utils_inductive import transform_dataset_by_idx
from models import HGAT
import sys
from print_log import Logger

logdir = "log/"
savedir = 'model/'
embdir = 'embeddings/'
makedirs([logdir, savedir, embdir])

write_embeddings = True
HOP = 3

dataset = 'example'

LR = 0.01 if dataset == 'snippets' else 0.005
DP = 0.3 if dataset in ['agnews', 'tagmynews'] else 0.3
WD = 0 if dataset == 'snippets' else 5e-6
LR = 0.05 if 'multi' in dataset else LR
DP = 0.5 if 'multi' in dataset else DP
WD = 0 if 'multi' in dataset else WD

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=LR,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=WD,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=DP,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--inductive', type=bool, default=True,
                    help='Whether use the transductive mode or inductive mode. ')
parser.add_argument('--dataset', type=str, default=dataset,
                    help='Dataset')
parser.add_argument('--repeat', type=int, default=20,
                    help='Number of repeated trials')
parser.add_argument('--node', action='store_false', default=True,
                    help='Use node-level attention or not. ')
parser.add_argument('--type', action='store_false', default=True,
                    help='Use type-level attention or not. ')

parser.add_argument('--out', default=f'outputs', help='directory to output the result')

# agnews 0.7 0.1 0.05 0.05
# mr 0.6 0.45 0.03 0.02
parser.add_argument('--no-uncertainty', action='store_true',
                    help='use uncertainty in the pesudo-label selection, default true')
parser.add_argument('--tau-p', default=0.70, type=float,
                    help='confidece threshold for positive pseudo-labels, default 0.70')
parser.add_argument('--tau-n', default=0.15, type=float,
                    help='confidece threshold for negative pseudo-labels, default 0.05')
parser.add_argument('--kappa-p', default=0.05, type=float,
                    help='uncertainty threshold for positive pseudo-labels, default 0.05')
parser.add_argument('--kappa-n', default=0.05, type=float,
                    help='uncertainty threshold for negative pseudo-labels, default 0.005')
parser.add_argument('--temp-nl', default=2.0, type=float,
                    help='temperature for generating negative pseduo-labels, default 2.0')
parser.add_argument('--no-progress', action='store_true',
                    help="don't use progress bar")
parser.add_argument('--class-blnc', default=10, type=int,
                    help='total number of class balanced iterations')

args = parser.parse_args()
dataset = args.dataset

args.cuda = not args.no_cuda and torch.cuda.is_available()

sys.stdout = Logger(logdir + "{}.log".format(dataset))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(1)
    print(torch.cuda.current_device())

loss_list = dict()


def margin_loss(preds, y, weighted_sample=False):
    nclass = y.shape[1]
    preds = preds[:, :nclass]
    y = y.float()
    lam = 0.25
    m = nn.Threshold(0., 0.)
    loss = y * m(0.9 - preds) ** 2 + \
           lam * (1.0 - y) * (m(preds - 0.1) ** 2)

    if weighted_sample:
        n, N = y.sum(dim=0, keepdim=True), y.shape[0]
        weight = torch.where(y == 1, n, torch.zeros_like(loss))
        weight = torch.where(y != 1, N - n, weight)
        weight = N / weight / 2
        loss = torch.mul(loss, weight)

    loss = torch.mean(torch.sum(loss, dim=1))
    return loss


def nll_loss(preds, y):
    y = y.max(1)[1].type_as(labels)
    return F.nll_loss(preds, y)


def evaluate(preds_list, y_list):
    nclass = y_list.shape[1]
    preds_list = preds_list[:, :nclass]
    if not preds_list.device == 'cpu':
        preds_list, y_list = preds_list.cpu(), y_list.cpu()

    threshold = 0.5
    multi_label = 'multi' in dataset
    if multi_label:
        y_list = y_list.numpy()
        preds_probs = preds_list.detach().numpy()
        preds = deepcopy(preds_probs)
        preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
        preds[np.where(preds >= threshold)] = 1.0
        preds[np.where(preds < threshold)] = 0.0
        [precision, recall, F1, support] = \
            precision_recall_fscore_support(y_list[preds.sum(axis=1) != 0], preds[preds.sum(axis=1) != 0],
                                            average='micro')
        [precision_ma, recall_ma, F1_ma, support] = \
            precision_recall_fscore_support(y_list[preds.sum(axis=1) != 0], preds[preds.sum(axis=1) != 0],
                                            average='macro')
        ER = accuracy_score(y_list, preds) * 100

        report = classification_report(y_list, preds, digits=5)

        print(' ER: %6.2f' % ER,
              'mi-ma: P: %5.1f %5.1f' % (precision * 100, precision_ma * 100),
              'R: %5.1f %5.1f' % (recall * 100, recall_ma * 100),
              'F1: %5.1f %5.1f' % (F1 * 100, F1_ma * 100),
              end="")
        return ER, report
    else:
        y_list = y_list.numpy()
        preds_probs = preds_list.detach().numpy()
        preds = deepcopy(preds_probs)
        preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
        preds[np.where(preds < 1)] = 0.0
        [precision, recall, F1, support] = \
            precision_recall_fscore_support(y_list, preds, average='macro')
        ER = accuracy_score(y_list, preds) * 100
        print(' Ac: %6.2f' % ER,
              'P: %5.1f' % (precision * 100),
              'R: %5.1f' % (recall * 100),
              'F1: %5.1f' % (F1 * 100),
              end="")
        return ER, F1


LOSS = margin_loss if 'multi' in dataset else nll_loss


def train(epoch,
          input_adj_train, input_features_train, idx_out_train, idx_train,
          input_adj_val, input_features_val, idx_out_val, idx_val):
    print('Epoch: {:04d}'.format(epoch + 1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(input_features_train, input_adj_train)

    if isinstance(output, list):
        O, L = output[0][idx_out_train], labels[idx_train]
    else:
        O, L = output[idx_out_train], labels[idx_train]
    loss_train = LOSS(O, L)
    print(' | loss: {:.4f}'.format(loss_train.item()), end='')
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        output = model(input_features_val, input_adj_val)
        if isinstance(output, list):
            loss_val = LOSS(output[0][idx_out_val], labels[idx_val])
            print(' | loss: {:.4f}'.format(loss_val.item()), end='')
            results = evaluate(output[0][idx_out_val], labels[idx_val])
        else:
            loss_val = LOSS(output[idx_out_val], labels[idx_val])
            print(' | loss: {:.4f}'.format(loss_val.item()), end='')
            results = evaluate(output[idx_out_val], labels[idx_val])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] = [loss_train.item()]

    if 'multi' in dataset:
        acc_val, res_line = results
        return float(acc_val.item()), res_line
    else:
        acc_val, f1_val = results
        return float(acc_val.item()), float(f1_val.item())


def train_with_nl_pl(epoch,
                     input_adj_train, input_features_train, idx_out_train, idx_train,
                     input_adj_val, input_features_val, idx_out_val, idx_val
                     , idx_out_nl, idx_nl, nl_labels):
    print('Epoch: {:04d}'.format(epoch + 1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(input_features_train, input_adj_train)

    loss_ce = 0
    loss_nl = 0

    loss_ce += F.cross_entropy(output[0][idx_out_train][:, :labels.shape[1]],
                               labels[idx_train].max(1)[1].type_as(labels), reduction='mean')

    if len(idx_out_nl) != 0:

        nl_logits = output[0][idx_out_nl][:, :labels.shape[1]]
        pre_nl = F.softmax(nl_logits, dim=1)
        pre_nl = 1 - pre_nl
        pre_nl = torch.clamp(pre_nl, 1e-7, 1.0)
        nl_mask = nl_labels[nl_idx]

        if args.no_cuda:
            y_nl = torch.ones((nl_logits.shape)).to(device='cuda', dtype=output[0].dtype)
        else:
            y_nl = torch.ones((nl_logits.shape)).to(device='cpu', dtype=output[0].dtype)
        loss_nl += torch.mean(
            (-torch.sum((y_nl * torch.log(pre_nl)) * nl_mask, dim=-1)) / (torch.sum(nl_mask, dim=-1) + 1e-7))

        loss_train = loss_ce + loss_nl
    else:
        loss_train = loss_ce

    print(' | loss: {:.4f}'.format(loss_train.item()), end='')

    loss_train.backward()
    optimizer.step()
    with torch.no_grad():
        model.eval()
        output = model(input_features_val, input_adj_val)
        if isinstance(output, list):
            loss_val = LOSS(output[0][idx_out_val], labels[idx_val])
            print(' | loss: {:.4f}'.format(loss_val.item()), end='')
            results = evaluate(output[0][idx_out_val], labels[idx_val])
        else:
            loss_val = LOSS(output[idx_out_val], labels[idx_val])
            print(' | loss: {:.4f}'.format(loss_val.item()), end='')
            results = evaluate(output[idx_out_val], labels[idx_val])
        print(' | time: {:.4f}s'.format(time.time() - t))
        loss_list[epoch] = [loss_train.item()]

    if 'multi' in dataset:
        acc_val, res_line = results
        return float(acc_val.item()), res_line
    else:
        acc_val, f1_val = results
        return float(acc_val.item()), float(f1_val.item())


def test(epoch, input_adj_test, input_features_test, idx_out_test, idx_test):
    with torch.no_grad():
        print(' ' * 90 if 'multi' in dataset else ' ' * 65, end='')
        t = time.time()
        model.eval()
        output = model(input_features_test, input_adj_test)

        if isinstance(output, list):
            loss_test = LOSS(output[0][idx_out_test], labels[idx_test])
            print(' | loss: {:.4f}'.format(loss_test.item()), end='')
            results = evaluate(output[0][idx_out_test], labels[idx_test])
        else:
            loss_test = LOSS(output[idx_out_test], labels[idx_test])
            print(' | loss: {:.4f}'.format(loss_test.item()), end='')
            results = evaluate(output[idx_out_test], labels[idx_test])
        print(' | time: {:.4f}s'.format(time.time() - t))
        loss_list[epoch] += [loss_test.item()]

    if 'multi' in dataset:
        acc_test, res_line = results
        return float(acc_test.item()), res_line
    else:
        acc_test, f1_test = results
        return float(acc_test.item()), float(f1_test.item())


FINAL_RESULT = []

BestACC = []
BESTF1 = []

All_RESULT = []
for iter in range(0, args.repeat):

    path = '../data/' + dataset + '/'
    adj, features, labels, idx_train_ori, idx_val_ori, idx_test_ori, idx_map = load_data(path=path, dataset=dataset)
    N = len(adj)

    # Transductive的数据集变换
    if args.inductive:
        print("Transfer to be inductive.")

        # resample
        # 第二轮直接从文件中读取
        if iter == 0:
            idx_train, idx_unlabeled, idx_val, idx_test = resample(args, idx_train_ori, idx_val_ori, idx_test_ori, path,
                                                                   idx_map, labels, iter)

            nl_labels = None
            nl_idx = None
        else:
            idx_train, idx_unlabeled, idx_val, idx_test, nl_idx, nl_labels = resample(args, idx_train_ori, idx_val_ori,
                                                                                      idx_test_ori, path, idx_map,
                                                                                      labels, iter)


        if iter == 0:
            input_adj_train, input_features_train, idx_related_train, idx_out_train = \
                transform_dataset_by_idx(adj, features, torch.cat([idx_train, idx_unlabeled]), idx_train, hop=HOP)
        elif nl_idx is not None :
            input_adj_train, input_features_train, idx_related_train, idx_out_train = \
                transform_dataset_by_idx(adj, features, torch.cat([idx_train, idx_unlabeled, nl_idx]), idx_train, hop=HOP)
        if nl_idx is None:
            input_adj_train, input_features_train, idx_related_train, idx_out_train = \
                transform_dataset_by_idx(adj, features, torch.cat([idx_train, idx_unlabeled]), idx_train, hop=HOP)


        input_adj_val, input_features_val, idx_related_val, idx_out_val = \
            transform_dataset_by_idx(adj, features, idx_val, idx_val, hop=HOP)

        input_adj_test, input_features_test, idx_related_test, idx_out_test = \
            transform_dataset_by_idx(adj, features, idx_test, idx_test, hop=HOP)

        input_adj_unlabel, input_features_unlabel, idx_related_unlabel, idx_out_unlabeled = \
            transform_dataset_by_idx(adj, features, idx_unlabeled, idx_unlabeled, hop=HOP)
        if nl_idx is not None:
            input_adj_nl, input_features_nl, idx_related_nl, idx_out_nl = \
                transform_dataset_by_idx(adj, features, nl_idx, nl_idx, hop=HOP)
        else:
            input_adj_nl, input_features_nl, idx_related_nl, idx_out_nl = \
                transform_dataset_by_idx(adj, features, idx_unlabeled, idx_unlabeled, hop=HOP)

        all_node_count = sum([_.shape[0] for _ in adj[0]])
        all_input_idx, all_related_idx = set(), set()

        for input_idx, related_idx in [[idx_train, idx_related_train],
                                       [idx_val, idx_related_val],
                                       [idx_test, idx_related_test],
                                       [idx_unlabeled, idx_related_unlabel]]:
            print("# input_nodes: {}, # related_nodes: {} / {}".format(
                len(input_idx), len(related_idx), all_node_count))
            all_input_idx.update(input_idx.numpy().tolist())
            all_related_idx.update(related_idx.numpy().tolist())
        print("Sum: # input_nodes: {}, # related_nodes: {} / {}\n".format(
            len(all_input_idx), len(all_related_idx), all_node_count))
    else:
        print("Transfer to be transductive.")
        idx_train, idx_unlabeled, idx_val, idx_test = resample(args, idx_train_ori, idx_val_ori, idx_test_ori, path,
                                                               idx_map, labels, iter)

        input_adj_train, input_features_train, idx_related_train, idx_out_train = \
            transform_dataset_by_idx(adj, features, idx_train, idx_train, hop=HOP)

        input_adj_val, input_features_val, idx_related_val, idx_out_val = \
            transform_dataset_by_idx(adj, features, idx_val, idx_val, hop=HOP)

        input_adj_test, input_features_test, idx_related_test, idx_out_test = \
            transform_dataset_by_idx(adj, features, idx_test, idx_test, hop=HOP)

        all_node_count = sum([_.shape[0] for _ in adj[0]])
        all_input_idx, all_related_idx = set(), set()

        for input_idx, related_idx in [[idx_train, idx_related_train],
                                       [idx_val, idx_related_val],
                                       [idx_test, idx_related_test]]:
            print("# input_nodes: {}, # related_nodes: {} / {}".format(
                len(input_idx), len(related_idx), all_node_count))
            all_input_idx.update(input_idx.numpy().tolist())
            all_related_idx.update(related_idx.numpy().tolist())
        print("Sum: # input_nodes: {}, # related_nodes: {} / {}\n".format(
            len(all_input_idx), len(all_related_idx), all_node_count))


    if args.cuda:
        N = len(features)
        for i in range(N):
            if input_features_train[i] is not None:
                input_features_train[i] = input_features_train[i].cuda()
            if input_features_val[i] is not None:
                input_features_val[i] = input_features_val[i].cuda()
            if input_features_test[i] is not None:
                input_features_test[i] = input_features_test[i].cuda()
            if args.inductive:
                if input_features_unlabel[i] is not None:
                    input_features_unlabel[i] = input_features_unlabel[i].cuda()
                if input_features_nl[i] is not None:
                    input_features_nl[i] = input_features_nl[i].cuda()
        for i in range(N):
            for j in range(N):
                if input_adj_train[i][j] is not None:
                    input_adj_train[i][j] = input_adj_train[i][j].cuda()
                if input_adj_val[i][j] is not None:
                    input_adj_val[i][j] = input_adj_val[i][j].cuda()
                if input_adj_test[i][j] is not None:
                    input_adj_test[i][j] = input_adj_test[i][j].cuda()
                if args.inductive:
                    if input_adj_unlabel[i][j] is not None:
                        input_adj_unlabel[i][j] = input_adj_unlabel[i][j].cuda()
                    if input_adj_nl[i][j] is not None:
                        input_adj_nl[i][j] = input_adj_nl[i][j].cuda()
        labels = labels.cuda()
        if args.inductive:
            if nl_labels is not None:
                nl_labels = nl_labels.cuda()

        idx_train, idx_out_train = idx_train.cuda(), idx_out_train.cuda()
        idx_val, idx_out_val = idx_val.cuda(), idx_out_val.cuda()
        idx_test, idx_out_test = idx_test.cuda(), idx_out_test.cuda()
        if args.inductive:
            idx_unlabeled, idx_out_unlabeled = idx_unlabeled.cuda(), idx_out_unlabeled.cuda()
            if nl_idx is not None:
                nl_idx, idx_out_nl = nl_idx.cuda(), idx_out_nl.cuda()

    # Model and optimizer
    print("\n\nNo. {} test.\n".format(iter + 1))
    model = HGAT(nfeat_list=[i.shape[1] for i in features],
                 type_attention=args.type,
                 node_attention=args.node,
                 nhid=args.hidden,
                 nclass=labels.shape[1],
                 dropout=args.dropout,
                 gamma=0.1,
                 orphan=True,
                 )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()

    resume_files = os.listdir(args.out)
    resume_itrs = [int(item.replace('.pth.tar', '').split("_")[-1]) for item in resume_files if
                   'checkpoint_iteration_' in item]
    if len(resume_itrs) > 0:
        checkpoint_itr = max(resume_itrs)
        resume_model = os.path.join(args.out, f'checkpoint_iteration_{checkpoint_itr}.pth.tar')
        if os.path.isfile(resume_model) and checkpoint_itr == iter:
            checkpoint = torch.load(resume_model)
            # best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    print(len(list(model.parameters())))
    print([i.size() for i in model.parameters()])
    # Train model
    t_total = time.time()
    vali_max = [0, [0, 0], -1, 0]

    test_accs = []
    test_f1s = []
    vali_accs = []
    vali_f1s = []

    for epoch in range(args.epochs):

        if iter == 0 or (not args.inductive) or nl_labels is None:

            vali_acc, vali_f1 = train(epoch,
                                      input_adj_train, input_features_train, idx_out_train, idx_train,
                                      input_adj_val, input_features_val, idx_out_val, idx_val)

        else:
            vali_acc, vali_f1 = train_with_nl_pl(epoch,
                                                 input_adj_train, input_features_train, idx_out_train, idx_train,
                                                 input_adj_val, input_features_val, idx_out_val, idx_val,
                                                 idx_out_nl, nl_idx, nl_labels)

        test_acc, test_f1 = test(epoch,
                                 input_adj_test, input_features_test, idx_out_test, idx_test)
        vali_accs.append(vali_acc)
        vali_f1s.append(vali_f1)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

        if test_acc > vali_max[1][0]:
            vali_max = [vali_acc, (test_acc, test_f1), epoch + 1, iter]

            with open(savedir + "{}.pkl".format(dataset), 'wb') as f:
                pkl.dump(model, f)

            if write_embeddings:
                makedirs([embdir])
                with open(embdir + "{}.emb".format(dataset), 'w') as f:
                    for i in model.emb.tolist():
                        f.write("{}\n".format(i))
                with open(embdir + "{}.emb2".format(dataset), 'w') as f:
                    for i in model.emb2.tolist():
                        f.write("{}\n".format(i))

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,

                'optimizer': optimizer.state_dict(),

            }, True, args.out, f'iteration_{str(iter)}')

    All_RESULT.append([iter, test_accs])
    All_RESULT.append([iter, vali_accs])

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if 'multi' in dataset:
        print("The best result is ACC: {0:.4f}, where epoch is {2}\n{1}\n".format(
            vali_max[1][0],
            vali_max[1][1],
            vali_max[2]))
    else:
        print("The best result is: ACC: {0:.4f} F1: {1:.4f}, where epoch is {2}\n\n".format(
            vali_max[1][0],
            vali_max[1][1],
            vali_max[2]))
        BestACC.append(vali_max[1][0])

    FINAL_RESULT.append(list(vali_max))

    # # trans to dataloader
    # dataset = OwnDataset( input_features_test, idx_out_test, idx_test)
    # unlbl_loader = DataLoader(dataset, batch_size=40, shuffle=False)

    checkpoint = torch.load(f'{args.out}/checkpoint_iteration_{str(iter)}.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.zero_grad()

    # pseudo-label generation and selection
    if args.inductive:
        pl_loss, pl_acc, pl_acc_pos, total_sel_pos, pl_acc_neg, total_sel_neg, unique_sel_neg, pseudo_label_dict = pseudo_labeling(
            args, input_adj_unlabel, input_features_unlabel, idx_out_unlabeled, idx_unlabeled, labels, model, iter)

        # 存伪标签的字典
        path1 = f'{args.out}/pseudo_labeling_iteration_{iter}.pkl'
        with open(path1, "wb") as f:
            pkl.dump(pseudo_label_dict, f)

        with open(os.path.join(args.out, 'log.txt'), 'a+') as ofile:
            ofile.write(f'############################# PL Iteration: {iter + 1} #############################\n')
            ofile.write(f'Last Test Acc: {test_acc}, Best Test Acc: {test_acc}\n')
            ofile.write(f'PL Acc (Positive): {pl_acc_pos}, Total Selected (Positive): {total_sel_pos}\n')
            ofile.write(
                f'PL Acc (Negative): {pl_acc_neg}, Total Selected (Negative): {total_sel_neg}, Unique Negative Samples: {unique_sel_neg}\n\n')


print("\n")
for i in range(len(FINAL_RESULT)):
    if 'multi' in dataset:
        print("{0}:\tvali:  {1:.5f}\ttest:  ACC: {2:.4f}, epoch={4}.\n{3}".format(
            i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][2]))
    else:
        print("{}:\tvali:  {:.5f}\ttest:  ACC: {:.4f} F1: {:.4f}, epoch={}".format(
            i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][2]))
for i in range(len(All_RESULT)):
    print(str(All_RESULT[i]) + '\n')
