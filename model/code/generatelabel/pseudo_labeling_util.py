
import time
import numpy as np
import torch
import torch.nn.functional as F
from .misc import AverageMeter
from .utils import enable_dropout



def pseudo_labeling(args, input_adj_test, input_features_test, idx_out_test, idx_test, labels, model, itr):

    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    pseudo_idx = []
    pseudo_target = []
    pseudo_maxstd = []
    gt_target = []
    idx_list = []
    gt_list = []
    target_list = []
    nl_mask = []
    model.eval()
    if not args.no_uncertainty:
        f_pass = 10
        enable_dropout(model)
    else:
        f_pass = 1



    with torch.no_grad():

        data_time.update(time.time() - end)
        inputs = input_features_test

        indexs = idx_test
        adj = input_adj_test
        targets = labels[indexs].max(1)[1].type_as(labels)
        idx_out_test = idx_out_test

        out_prob = []
        out_prob_nl = []

        for _ in range(f_pass):
            outputs = model(inputs, adj)
            outputs = outputs[0][idx_out_test][:, :labels.shape[1]]
            out_prob.append(F.softmax(outputs, dim=1))  # for selecting positive pseudo-labels
            out_prob_nl.append(F.softmax(outputs / args.temp_nl, dim=1))  # for selecting negative pseudo-labels

        out_prob = torch.stack(out_prob)
        out_prob_nl = torch.stack(out_prob_nl)
        out_std = torch.std(out_prob, dim=0)
        out_std_nl = torch.std(out_prob_nl, dim=0)
        out_prob = torch.mean(out_prob, dim=0)
        out_prob_nl = torch.mean(out_prob_nl, dim=0)
        max_value, max_idx = torch.max(out_prob, dim=1)
        max_std = out_std.gather(1, max_idx.view(-1, 1))
        out_std_nl = out_std_nl.cpu().numpy()

        # selecting negative pseudo-labels
        interm_nl_mask = ((out_std_nl < args.kappa_n) * (out_prob_nl.cpu().numpy() < args.tau_n)) * 1

        # manually setting the argmax value to zero
        for enum, item in enumerate(max_idx.cpu().numpy()):
            interm_nl_mask[enum, item] = 0
        nl_mask.extend(interm_nl_mask)
        # nl_mask 表示负样本
        idx_list.extend(idx_test.cpu().numpy().tolist())
        gt_list.extend(targets.cpu().numpy().tolist())
        target_list.extend(max_idx.cpu().numpy().tolist())

        # selecting positive pseudo-labels
        if not args.no_uncertainty:
            selected_idx = (max_value >= args.tau_p) * (max_std.squeeze(1) < args.kappa_p)
        else:
            selected_idx = max_value >= args.tau_p

        pseudo_maxstd.extend(max_std.squeeze(1)[selected_idx].cpu().numpy().tolist())
        pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
        pseudo_idx.extend(idx_test[selected_idx].cpu().numpy().tolist())
        gt_target.extend(targets[selected_idx].cpu().numpy().tolist())

    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    pseudo_maxstd = np.array(pseudo_maxstd)
    pseudo_idx = np.array(pseudo_idx)

    # class balance the selected pseudo-labels
    if len(pseudo_target) > 0:
        min_count = 5000000  # arbitary large value
        for class_idx in range(labels.shape[1]):
            class_len = len(np.where(pseudo_target == class_idx)[0])
            if class_len < min_count:
                min_count = class_len
        min_count = max(25,
                        min_count)  # this 25 is used to avoid degenarate cases when the minimum count for a certain class is very low

        blnc_idx_list = []
        for class_idx in range(labels.shape[1]):
            current_class_idx = np.where(pseudo_target == class_idx)
            if len(np.where(pseudo_target == class_idx)[0]) > 0:
                current_class_maxstd = pseudo_maxstd[current_class_idx]
                sorted_maxstd_idx = np.argsort(current_class_maxstd)
                current_class_idx = current_class_idx[0][
                    sorted_maxstd_idx[:min_count]]  # select the samples with lowest uncertainty
                blnc_idx_list.extend(current_class_idx)

        blnc_idx_list = np.array(blnc_idx_list)
        pseudo_target = pseudo_target[blnc_idx_list]
        pseudo_idx = pseudo_idx[blnc_idx_list]
        gt_target = gt_target[blnc_idx_list]
    pseudo_labeling_acc = (pseudo_target == gt_target)*1
    if len(pseudo_labeling_acc) > 0:
        pseudo_labeling_acc = (sum(pseudo_labeling_acc)/len(pseudo_labeling_acc))*100
        print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')

    pseudo_nl_mask = []
    pseudo_nl_idx = []
    nl_gt_list = []

    for i in range(len(idx_list)):
        if idx_list[i] not in pseudo_idx and sum(nl_mask[i]) > 0:
            pseudo_nl_mask.append(nl_mask[i])
            pseudo_nl_idx.append(idx_list[i])
            nl_gt_list.append(gt_list[i])


    if len(nl_gt_list) != 0:
        nl_gt_list = np.array(nl_gt_list)
        pseudo_nl_mask = np.array(pseudo_nl_mask)
        one_hot_targets = np.eye(labels.shape[1])[nl_gt_list]
        one_hot_targets = one_hot_targets - 1
        one_hot_targets = np.abs(one_hot_targets)
        flat_pseudo_nl_mask = pseudo_nl_mask.reshape(1, -1)[0]
        flat_one_hot_targets = one_hot_targets.reshape(1, -1)[0]
        flat_one_hot_targets = flat_one_hot_targets[np.where(flat_pseudo_nl_mask == 1)]
        flat_pseudo_nl_mask = flat_pseudo_nl_mask[np.where(flat_pseudo_nl_mask == 1)]
        nl_accuracy = (flat_pseudo_nl_mask == flat_one_hot_targets) * 1
        nl_accuracy_final = (sum(nl_accuracy) / len(nl_accuracy)) * 100
        print(
            f'Pseudo-Labeling Accuracy (negative): {nl_accuracy_final}, Total Selected: {len(nl_accuracy)}, Unique Samples: {len(pseudo_nl_mask)}')
        pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target': pseudo_target.tolist(),
                         'nl_idx': pseudo_nl_idx, 'nl_mask': pseudo_nl_mask.tolist()}
    else:
        pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target': pseudo_target.tolist(),
                             'nl_idx': [], 'nl_mask': []}
        nl_accuracy_final = 0
        nl_accuracy = []
        pseudo_nl_mask =[]


    return losses.avg, top1.avg, pseudo_labeling_acc, len(pseudo_idx), nl_accuracy_final, len(nl_accuracy), len(
        pseudo_nl_mask), pseudo_label_dict
