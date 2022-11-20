import torch

import argparse
import os
import random
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset.dataset import getIMDB
from dataset.dataset import IMDBDataset

from model.classifier import Classifier

from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.dataset import collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imdb')
parser.add_argument('--data_path', type=str, default='../dataset/')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs in training')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--lrbase', default=0.00001, type=float, help='initial learning rate for base model')
parser.add_argument('--lrlast', default=0.001, type=float, help='initial learning rate for classifier')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--alpha', default=0.75, type=float, help='alpha for beta distribution')
parser.add_argument('--beta', default=-1, type=float, help='another param for beta distribution')
parser.add_argument('--tau', default=1, type=float, help='tau for dirichlet distribution')
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--out_att', action='store_true', help='output attentions')

parser.add_argument('--n_sample', type=int, default=2, help='num of aug samples')

parser.add_argument('--data_size', type=int, default=0, help="size of training data in low-resource settings")

parser.add_argument('--model', default='bert', type=str, help='baseline model')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default: 0.8)')
parser.add_argument('--save_dir', type=str, default='model_logs/', help='dir for saving models')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.use_cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()

print("gpu num: ", args.n_gpu)
print("dataset: ", args.dataset)
print("model:", args.model)
print("seed: ", args.seed)

print("lrbase: ", args.lrbase)
print("batch size: ", args.batch_size)

print("num of samples: ", args.n_sample)

print("output_attentions: ", args.out_att)

best_acc = 0
best_f1 = 0


# best_loss = float('inf')
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpu) > 0:
        torch.cuda.manual_seed_all(args.seed)


# def train(train_loader, model, optimizer, criterion, num_label=3):
#     model.train()
#     pbar = tqdm(train_loader)
#
#     for batch_idx, batch in enumerate(pbar):
#         inputs, targets = batch['inputs'], batch['labels']
#         targets = targets.cuda(non_blocking=True)
#
#         for k, v in inputs.items(): inputs[k] = v.cuda()
#
#         if args.mix_option:
#             idx = torch.randperm(inputs['input_ids'].size(0))  # mix training examples in each batch
#             inputs_2 = {}
#             for k, v in inputs.items():
#                 inputs_2[k] = v[idx]
#             targets_2 = targets[idx]
#
#             mix_layer = np.random.choice(args.mix_layer_set, 1)[0] - 1
#             if (args.alpha) < 1e-6:
#                 lam = 1
#             else:
#                 if args.beta == -1:
#                     lam = np.random.beta(args.alpha, args.alpha)
#                 else:
#                     lam = np.random.beta(args.alpha, args.beta)
#                 lam = max(lam, 1 - lam)
#             mix_outputs, _ = model(inputs, inputs_2, lam=lam, mix_layer=mix_layer, mix_option=args.mix_option)
#             if args.tmix:
#                 targets_temp1 = torch.zeros(inputs['input_ids'].size(0), num_label).cuda()
#                 targets_onehot1 = targets_temp1.scatter_(1, targets.unsqueeze(1), 1)
#                 targets_temp2 = torch.zeros(inputs_2['input_ids'].size(0), num_label).cuda()
#                 targets_onehot2 = targets_temp2.scatter_(1, targets_2.unsqueeze(1), 1)
#                 mix_targets = lam * targets_onehot1 + (1 - lam) * targets_onehot2
#                 loss = -torch.mean(torch.sum(F.log_softmax(mix_outputs, dim=1) * mix_targets, dim=1))
#             else:
#                 loss_func = mix_criterion(targets, targets_2, lam)
#                 loss = loss_func(criterion, mix_outputs)
#             pbar.set_description('loss={:.4f}'.format(loss))
#         else:
#             outputs, _ = model(inputs)
#             loss = criterion(outputs, targets)
#             pbar.set_description('loss={:.4f}'.format(loss))
#
#         if args.n_gpu > 1:
#             loss = loss.mean()
#
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#         optimizer.step()


def train_aug(train_loader, model, optimizer):
    model.train()

    pbar = tqdm(train_loader)
    for batch_idx, batch in enumerate(pbar):
        orig_inputs, aug_inputs, targets = batch['ori_inputs'], batch['aug_inputs'], batch['labels']
        for k, v in orig_inputs.items(): orig_inputs[k] = v.cuda()
        for k, v in aug_inputs.items(): aug_inputs[k] = v.cuda()

        if (args.alpha) < 1e-6:
            lam = 1
        else:
            if args.beta == -1:
                lam = np.random.beta(args.alpha, args.alpha)
            else:
                lam = np.random.beta(args.alpha, args.beta)
            lam = max(lam, 1 - lam)
        ws = np.random.dirichlet([args.tau] * args.n_sample)

        loss, predict, _ = model([orig_inputs['input_ids'], aug_inputs['input_ids']],
                                 [orig_inputs['attention_mask'], aug_inputs['attention_mask']],
                                 [orig_inputs['token_type_ids'], aug_inputs['token_type_ids']],
                                 [targets, targets]
                                 )

        # if args.no_jsd:
        #     loss = criterion(mix_outputs, targets.cuda())
        # else:  # JSD
        #     orig_outputs, _ = model(orig_inputs)
        #     loss = criterion(orig_outputs, targets.cuda())
        #     p_clean = F.softmax(orig_outputs, dim=1)
        #     p_aug = F.softmax(mix_outputs, dim=1)
        #     p_mixture = torch.clamp((p_clean + p_aug) / 2., 1e-7, 1).log()
        #     loss += 8 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
        #                  F.kl_div(p_mixture, p_aug, reduction='batchmean')) / 2.  # coefficient = 8

        if args.n_gpu > 1:
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        pbar.set_description('loss={:4f}'.format(loss))


def validate(val_loader, model):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        preds = np.array([], dtype=int)
        labels = np.array([], dtype=int)

        for batch_idx, batch in enumerate(val_loader):
            inputs, targets = batch['inputs'], batch['labels']
            for k, v in inputs.items(): inputs[k] = v.cuda()
            batch_size = inputs['input_ids'].size(0)

            targets = targets.cuda(non_blocking=True)

            loss, predict, outputs = model([inputs['input_ids']],
                                           [inputs['attention_mask']],
                                           [inputs['token_type_ids']],
                                           [targets]
                                           )

            correct += (np.array(predict.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * batch_size
            total_sample += batch_size
            preds = np.append(preds, np.array(predict.cpu()))
            labels = np.append(labels, np.array(targets.cpu()))

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample
        # if num_label[args.dataset] == 2:
        f1_total = f1_score(labels, preds, average='binary')
        # else:
        #     f1_total = f1_score(labels, preds, average='macro')

    return loss_total, acc_total, f1_total


if __name__ == "__main__":

    set_seed(args)

    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = Classifier()
    else:
        raise NotImplementedError

    train_data, dev_data, test_data = getIMDB()

    train_set = IMDBDataset(train_data, train=True)
    dev_set = IMDBDataset(dev_data)
    test_set = IMDBDataset(test_data)

    train_loader = Data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, True))
    val_loader = Data.DataLoader(
        dataset=dev_set, batch_size=64, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=64, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))

    # print("loading model...")
    # model.load_state_dict(torch.load("model_logs/snli1_model.pt"))
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        optimizer = AdamW(
            [
                {"params": model.module.bert.parameters(), "lr": args.lrbase},
                {"params": model.module.dense.parameters(), "lr": args.lrlast},
            ])
    else:
        optimizer = AdamW(
            [
                {"params": model.bert.parameters(), "lr": args.lrbase},
                {"params": model.dense.parameters(), "lr": args.lrlast},
            ])

    test_accs = []
    test_f1s = []

    for epoch in range(args.epochs):

        train_aug(train_loader, model, optimizer)

        val_loss, val_acc, val_f1 = validate(val_loader, model)

        print("epoch {}, val acc {}, val f1 {}, val_loss {}".format(
            epoch, val_acc, val_f1, val_loss))
        # val_losses.append(val_loss)

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc, test_f1 = validate(test_loader, model)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            torch.save(model.state_dict(), "model.pt")
            print("epoch {}, test acc {}, test f1 {}, test loss {}".format(
                epoch, test_acc, test_f1, test_loss))

        if val_f1 >= best_f1:
            best_f1 = val_f1
            test_loss, test_acc, test_f1 = validate(test_loader, model)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            torch.save(model.state_dict(), "model.pt")
            print("epoch {}, test acc {}, test f1 {}, test loss {}".format(
                epoch, test_acc, test_f1, test_loss))

    print('Best test acc:')
    print(max(test_accs))

    print('Best test f1')
    print(max(test_f1s))
