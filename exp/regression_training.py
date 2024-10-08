"""
classical single-step stock forecasting task
follow HIST repo
"""
import torch
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, sys.path[0]+"/../")
from models.model import MLP, HIST, GRU, LSTM, GAT, RSR
from utils.utils import metric_fn, mse, loss_ic, pair_wise_loss, NDCG_loss, ApproxNDCG_loss
from utils.dataloader import create_mto_loaders
import warnings
import logging


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12
warnings.filterwarnings('ignore')


def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    raise ValueError('unknown model name `%s`'%model_name)


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    # output is a single dim tensor
    return mse(pred[mask], label[mask])


global_log_file = None


def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1


def train_epoch(epoch, model, optimizer, train_loader, writer, args,
                stock2concept_matrix=None, stock2stock_matrix = None):
    """
    train epoch function
    :param epoch: number of epoch of training
    :param model: model that will be used
    :param optimizer:
    :param train_loader:
    :param writer:
    :param args:
    :param stock2concept_matrix:
    :return:
    """

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, _, label, market_value, stock_index, _, mask = train_loader.get(slc)
        # feature here is the data batch_size, 360
        if args.model_name == 'HIST':
            # if HIST is used, take the stock2concept_matrix and market_value
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
            # the stock2concept_matrix[stock_index] is a matrix, shape is (the number of stock index, predefined con)
            # the stock2concept_matrix has been sort to the order of stock index
        elif args.model_name == 'RSR':
            pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
        else:
            # other model only use feature as input
            pred = model(feature)
        if args.loss_type == 'ic':
            loss = loss_ic(pred, label)
        elif args.loss_type == 'pair_wise':
            loss = pair_wise_loss(args, pred, label)
        elif args.loss_type == 'ndcg':
            loss = NDCG_loss(pred, label)
        elif args.loss_type == 'appndcg':
            loss = ApproxNDCG_loss(pred, label)
        else:
            loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, stock2stock_matrix=None, prefix='Test'):
    """
    :return: loss -> mse
             scores -> ic
             rank_ic
             precision, recall -> precision, recall @1, 3, 5, 10, 20, 30, 50, 100
    """

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, _, label, market_value, stock_index, index, mask = test_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                pred = model(feature)

            if args.loss_type == 'pair_wise':
                loss = pair_wise_loss(args, pred, label)
            elif args.loss_type == 'appndcg':
                loss = ApproxNDCG_loss(pred, label)
            else:
                loss = loss_fn(pred, label)
            preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    # evaluate
    preds = pd.concat(preds, axis=0)
    # use metric_fn to compute precision, recall, ic and rank ic
    precision, recall, ic, rank_ic, ndcg = metric_fn(preds)
    scores = rank_ic

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic, ndcg


def inference(model, data_loader, stock2concept_matrix=None, stock2stock_matrix=None):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, _, label, market_value, stock_index, index, mask = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif args.model_name == 'RSR':
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            else:
                pred = model(feature)
            preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),}, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = '../ouput/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_mto_loaders(args, device=device)

    stock2concept_matrix = np.load(args.stock2concept_matrix)
    stock2stock_matrix = np.load(args.stock2stock_matrix)
    if args.model_name == 'HIST':
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    if args.model_name == 'RSR':
        stock2stock_matrix = torch.Tensor(stock2stock_matrix).to(device)
        num_relation = stock2stock_matrix.shape[2]

    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    all_ndcg = []
    global_best_score = -np.inf
    for times in range(args.repeat):
        pprint('create model...')
        if args.model_name == 'HIST':
            # model = get_model(args.model_name)(d_feat=args.d_feat, num_layers=args.num_layers, K=args.K)
            model = get_model(args.model_name)(args)
        elif args.model_name == 'RSR':
            model = get_model(args.model_name)(args, num_relation=num_relation)
        else:
            model = get_model(args.model_name)(args, d_feat=args.d_feat, num_layers=args.num_layers)
        
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        # save best parameters
        best_param = copy.deepcopy(model.state_dict())
        for epoch in range(args.n_epochs):
            pprint('Running', times, 'Epoch:', epoch)

            pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix, stock2stock_matrix)
            # save model  after every epoch
            # -------------------------------------------------------------------------

            params_ckpt = copy.deepcopy(model.state_dict())
            pprint('evaluating...')
            # compute the loss, score, pre, recall, ic, rank_ic on train, valid and test data
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic, train_ndcg = test_epoch(epoch, model, train_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Train')
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic, val_ndcg = test_epoch(epoch, model, valid_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Valid')
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic, test_ndcg = test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix, stock2stock_matrix, prefix='Test')

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, val_ic, test_ic))
            pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (train_rank_ic, val_rank_ic, test_rank_ic))
            pprint('Train Precision: ', train_precision)
            pprint('Valid Precision: ', val_precision)
            pprint('Test Precision: ', test_precision)
            pprint('Train Recall: ', train_recall)
            pprint('Valid Recall: ', val_recall)
            pprint('Test Recall: ', test_recall)
            pprint('Train NDCG: ', train_ndcg)
            pprint('Valid NDCG: ', val_ndcg)
            pprint('Test NDCG: ', test_ndcg)

            if val_score > best_score:
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(params_ckpt)
            else:
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, output_path+'/model.bin')
        if best_score > global_best_score:
            torch.save(best_param, output_path + '/model_best.bin')
            global_best_score = best_score

        pprint('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:
            # do prediction on train, valid and test data
            pred = inference(model, eval(name+'_loader'), stock2concept_matrix=stock2concept_matrix,
                             stock2stock_matrix=stock2stock_matrix)
            # pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

            precision, recall, ic, rank_ic, ndcg = metric_fn(pred)

            pprint('%s: IC %.6f Rank IC %.6f' % (name, ic.mean(), rank_ic.mean()))
            pprint(name, ': Precision ', precision)
            pprint(name, ': Recall ', recall)
            pprint(name, ':NDCG', ndcg)
            res[name+'-IC'] = ic
            res[name+'-RankIC'] = rank_ic
            res[name+'-NDCG@100'] = ndcg[100]
        
        all_precision.append(list(precision.values()))
        all_recall.append(list(recall.values()))
        all_ndcg.append(list(ndcg.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/'+key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path+'/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)
    pprint('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)' % (np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    precision_mean = np.array(all_precision).mean(axis=0)
    precision_std = np.array(all_precision).std(axis=0)
    ndcg_mean = np.array(all_ndcg).mean(axis=0)
    ndcg_std = np.array(all_ndcg).std(axis=0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        pprint('Precision@%d: %.4f (%.4f)' % (N[k], precision_mean[k], precision_std[k]))
        pprint('NDCG@%d: %.4f (%.4f)' % (N[k], ndcg_mean[k], ndcg_std[k]))

    pprint('finished.')


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model setting
    parser.add_argument('--model_name', default='HIST')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--loss_type', default='')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--task_name', type=str, default='regression', help='task setup')

    # training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--repeat', type=int, default=3)

    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1)  # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='')  # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')

    # other
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='PatchTST')

    # input for csi 300
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to22.pkl')
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock2stock_matrix', default='./data/csi300_multi_stock2stock.npy')
    parser.add_argument('--mtm_source_path', default='./data/original_mtm.pkl')
    parser.add_argument('--mtm_column', default='mtm0604')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')
    parser.add_argument('--outdir', default='./output/csi300_RSR')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args)
