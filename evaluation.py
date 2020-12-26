""" This module is for model evaluation with filtering. """
import logging

import numpy as np
import torch
from torch.nn import functional as F


log = logging.getLogger()


def ranking_and_hits(model, dev_rank_batcher, batch_size, name, silent=False):
    print_info(name)

    # initial
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        rel_reverse = str2var['rel_eval']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()

        pred1_ = model.forward(e1, rel)
        pred2_ = model.forward(e2, rel_reverse)

        # sigmoid is applied in loss function
        # here we do it manually to get positive predictions
        pred1 = F.sigmoid(pred1_)
        pred2 = F.sigmoid(pred2_)

        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(batch_size):
            filter1 = e2_multi1[i][e2_multi1[i] != -1].long()
            filter2 = e2_multi2[i][e2_multi2[i] != -1].long()

            # save the prediction that is relevant
            target_value1 = pred1[i, e2[i, 0]].item()
            target_value2 = pred2[i, e1[i, 0]].item()

            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(batch_size):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i] == e2[i, 0].item())[0]
            rank2 = np.where(argsort2[i] == e1[i, 0].item())[0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)

    hits10 = [1 if x <= 10 else 0 for x in ranks]
    hits3 = [1 if x <= 5 else 0 for x in ranks]
    hits1 = [1 if x == 1 else 0 for x in ranks]
    mr = np.mean(ranks)
    mrr = np.mean(1. / np.array(ranks))

    if not silent:
        log.info('Hits @10: %f', np.mean(hits10))
        log.info('Hits @3: %f', np.mean(hits3))
        log.info('Hits @1: %f', np.mean(hits1))
        log.info('Mean rank: %f', mr)
        log.info('Mean reciprocal rank: %f', mrr)

    return mr


def print_info(name):
    log.info('')
    log.info('-' * 50)
    log.info(name)
    log.info('-' * 50)
    log.info('')