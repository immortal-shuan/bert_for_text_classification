import os
import torch
import random
import numpy as np
from adv import FGM, PGD
from torch.optim import Adam
from tqdm import tqdm, trange
from arg import init_arg_parser
from sklearn.model_selection import KFold
from model.bert_model import bert_classifi
from data_process import data_process, Batch
from sklearn.metrics import precision_score, recall_score, f1_score


def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def train_trail(model, optimizer, criteria, args):

    train_ids, dev_ids, dev_label_num = data_process(args)
    random.shuffle(train_ids)

    train_num = len(train_ids)
    model.zero_grad()

    dev_pre = 0.0
    max_pre_index = 0

    dev_rec = 0.0
    max_rec_index = 0

    dev_f1 = 0.0
    max_f1_index = 0

    if args.use_fgm:
        fgm = FGM(model)
    if args.use_pdg:
        pgd = PGD(model)

    for i in range(args.epoch_num):

        train_step = 1.0
        train_loss = 0.0

        train_preds = []
        train_labels = []

        for j in trange(0, train_num, args.batch_size):
            model.train()
            if j + args.batch_size < train_num:
                train_batch_data = train_ids[j:j + args.batch_size]
            else:
                train_batch_data = train_ids[j:train_num]
            text_ids, text_type, text_mask, label = Batch(train_batch_data, args)

            out = model(text_ids, text_type, text_mask)

            loss = criteria(out, label)
            train_loss += loss.item()

            loss = loss / args.loss_step
            loss.backward()

            if args.use_fgm:
                fgm.attack()
                out_adv = model(text_ids, text_type, text_mask)
                loss_adv = criteria(out_adv, label)
                loss_adv = loss_adv / args.loss_step
                loss_adv.backward()
                fgm.restore()

            if args.use_pdg:
                pgd.backup_grad()
                for t in range(args.k_pdg):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != args.k_pdg - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    out_adv = model(text_ids, text_type, text_mask)
                    loss_adv = criteria(out_adv, label)
                    loss_adv = loss_adv / args.loss_step
                    loss_adv.backward()
                pgd.restore()

            if int(train_step % args.loss_step) == 1:
                optimizer.step()
                model.zero_grad()

            pred = out.argmax(-1).cpu().tolist()
            train_preds.extend(pred)
            if args.use_label_smoothing:
                train_labels.extend(label.argmax(-1).cpu().tolist())
            else:
                train_labels.extend(label.cpu().tolist())
            train_step += 1.0

        train_f1 = f1_score(np.array(train_labels), np.array(train_preds), average='macro')

        print('epoch:{}\n train_loss:{}\n train_f1:{}'.format(i, train_loss / train_step, train_f1))

        dev_pre_, dev_rec_, dev_f1_ = dev(model=model, args=args, dev_data=dev_ids)

        if dev_pre < dev_pre_:
            dev_pre = dev_pre_
            max_pre_index = i

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}.pth'.format(i))
                torch.save(model.state_dict(), save_file)

        if dev_rec < dev_rec_:
            dev_rec = dev_rec_
            max_rec_index = i

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}.pth'.format(i))
                if not os.path.exists(save_file):
                    torch.save(model.state_dict(), save_file)

        if dev_f1 < dev_f1_:
            dev_f1 = dev_f1_
            max_f1_index = i

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}.pth'.format(i))
                if not os.path.exists(save_file):
                    torch.save(model.state_dict(), save_file)

        if i - max_f1_index > args.stop_num:
            break

    file = open('result.txt', 'a')
    file.write('max_pre: {}, {}'.format(max_pre_index, dev_pre) + '\n')
    file.write('max_rec: {}, {}'.format(max_rec_index, dev_rec) + '\n')
    file.write('max_f1: {}, {}'.format(max_f1_index, dev_f1) + '\n' + '\n')
    file.close()

    print('-----------------------------------------------------------------------------------------------------------')
    print('max_pre: {}, {}'.format(max_pre_index, dev_pre))
    print('max_rec: {}, {}'.format(max_rec_index, dev_rec))
    print('max_f1: {}, {}'.format(max_f1_index, dev_f1))
    print('-----------------------------------------------------------------------------------------------------------')

    if args.out_dev_result:
        model = bert_classifi(args=args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        the_best_f1_model = torch.load(os.path.join(args.output_path, 'model_{}.pth'.format(max_f1_index)))
        model.load_state_dict(the_best_f1_model)
        model = model.to(device)
        dev_result_info(model=model, args=args, dev_data=dev_ids, dev_label_num=dev_label_num)


def dev(model, args, dev_data):
    model.eval()

    dev_len = len(dev_data)

    dev_preds = []
    dev_labels = []

    with torch.no_grad():
        for m in trange(0, dev_len, args.batch_size):
            if m + args.batch_size < dev_len:
                dev_batch_data = dev_data[m:m + args.batch_size]
            else:
                dev_batch_data = dev_data[m:dev_len]
            text_input_, text_type, text_mask, label = Batch(dev_batch_data, args)

            out = model(text_input_, text_type, text_mask)

            pred = out.argmax(-1).cpu().tolist()
            dev_preds.extend(pred)
            if args.use_label_smoothing:
                dev_labels.extend(label.argmax(-1).cpu().tolist())
            else:
                dev_labels.extend(label.cpu().tolist())

    dev_pre = precision_score(np.array(dev_labels), np.array(dev_preds), average='macro')
    dev_rec = recall_score(np.array(dev_labels), np.array(dev_preds), average='macro')
    dev_f1 = f1_score(np.array(dev_labels), np.array(dev_preds), average='macro')

    print('dev_pre:{}, dev_rec:{}, dev_f1:{}'.format(dev_pre, dev_rec, dev_f1))
    return dev_pre, dev_rec, dev_f1


def dev_result_info(model, args, dev_data, dev_label_num):
    assert len(dev_data) == sum(dev_label_num)

    model.eval()

    dev_len = len(dev_data)

    dev_preds = []
    dev_labels = []
    with torch.no_grad():
        for m in trange(0, dev_len, args.batch_size):
            if m + args.batch_size < dev_len:
                dev_batch_data = dev_data[m:m + args.batch_size]
            else:
                dev_batch_data = dev_data[m:dev_len]
            text_input_, text_type, text_mask, label = Batch(dev_batch_data, args)

            out = model(text_input_, text_type, text_mask)

            pred = out.argmax(-1).cpu().tolist()
            dev_preds.extend(pred)
            if args.use_label_smoothing:
                dev_labels.extend(label.argmax(-1).cpu().tolist())
            else:
                dev_labels.extend(label.cpu().tolist())

    dev_pre = precision_score(np.array(dev_labels), np.array(dev_preds), average=None)
    dev_rec = recall_score(np.array(dev_labels), np.array(dev_preds), average=None)
    dev_f1 = f1_score(np.array(dev_labels), np.array(dev_preds), average=None)

    print('dev_pre:{}'.format(dev_pre))
    print('dev_rec:{}'.format(dev_rec))
    print('dev_f1:{}'.format(dev_f1))

    dev_label_score = []
    for i in range(args.class_num):
        sub_preds = dev_preds[sum(dev_label_num[:i]): sum(dev_label_num[:i+1])]
        sub_labels = dev_labels[sum(dev_label_num[:i]): sum(dev_label_num[:i+1])]

        assert sub_labels == [i] * dev_label_num[i]

        sub_preds = (np.array(sub_preds, dtype=int) == int(i)) + 0
        sub_labels = (np.array(sub_labels, dtype=int) == int(i)) + 0

        dev_pre = precision_score(sub_labels, sub_preds)
        dev_rec = recall_score(sub_labels, sub_preds)
        dev_f1 = f1_score(sub_labels, sub_preds)
        dev_label_score.append(['{}'.format(i), dev_pre, dev_rec, dev_f1])
    print(dev_label_score)


def cross_train(args):
    train_ids = data_process(args=args)

    kfold = KFold(n_splits=args.k, shuffle=True)
    index = kfold.split(train_ids)

    k = 1
    for train_index, dev_index in index:
        train_data_ = np.array(train_ids)[train_index].tolist()
        dev_data = np.array(train_ids)[dev_index].tolist()

        model = bert_classifi(args=args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        fc_para = list(map(id, model.fc.parameters()))
        base_para = filter(lambda p: id(p) not in fc_para, model.parameters())
        params = [{'params': base_para},
                  {'params': model.fc.parameters(), 'lr': args.fc_lr}]

        optimizer = Adam(params, lr=args.bert_lr, amsgrad=args.use_amsgrad)

        if args.use_label_smoothing:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        train(model=model, train_ids=train_data_, dev_ids=dev_data, optimizer=optimizer, criteria=criterion, k=k, args=args)
        k = k + 1


def train(model, train_ids, dev_ids, optimizer, criteria, k, args):
    train_num = len(train_ids)
    model.zero_grad()

    dev_pre = 0.0
    max_pre_index = 0

    dev_rec = 0.0
    max_rec_index = 0

    dev_f1 = 0.0
    max_f1_index = 0

    if args.use_fgm:
        fgm = FGM(model)
    if args.use_pdg:
        pgd = PGD(model)

    for i in range(args.epoch_num):

        train_step = 1.0
        train_loss = 0.0

        train_preds = []
        train_labels = []

        for j in trange(0, train_num, args.batch_size):
            model.train()
            if j + args.batch_size < train_num:
                train_batch_data = train_ids[j:j + args.batch_size]
            else:
                train_batch_data = train_ids[j:train_num]
            text_ids, text_type, text_mask, label = Batch(train_batch_data, args)

            out = model(text_ids, text_type, text_mask)

            loss = criteria(out, label)
            train_loss += loss.item()

            loss = loss / args.loss_step
            loss.backward()

            if args.use_fgm:
                fgm.attack()
                out_adv = model(text_ids, text_type, text_mask)
                loss_adv = criteria(out_adv, label)
                loss_adv = loss_adv / args.loss_step
                loss_adv.backward()
                fgm.restore()

            if args.use_pdg:
                pgd.backup_grad()
                for t in range(args.k_pdg):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != args.k_pdg - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    out_adv = model(text_ids, text_type, text_mask)
                    loss_adv = criteria(out_adv, label)
                    loss_adv = loss_adv / args.loss_step
                    loss_adv.backward()
                pgd.restore()

            if int(train_step % args.loss_step) == 0:
                optimizer.step()
                model.zero_grad()

            pred = out.argmax(-1).cpu().tolist()
            train_preds.extend(pred)
            if args.use_label_smoothing:
                train_labels.extend(label.argmax(-1).cpu().tolist())
            else:
                train_labels.extend(label.cpu().tolist())
            train_step += 1.0

        train_f1 = f1_score(np.array(train_labels), np.array(train_preds), average='macro')

        print('epoch:{}\n train_loss:{}\n train_f1:{}'.format(i, train_loss / train_step, train_f1))

        dev_pre_, dev_rec_, dev_f1_ = dev(model=model, args=args, dev_data=dev_ids)

        if dev_pre < dev_pre_:
            dev_pre = dev_pre_
            max_pre_index = i

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}_{}.pth'.format(k, i))
                torch.save(model.state_dict(), save_file)

        if dev_rec < dev_rec_:
            dev_rec = dev_rec_
            max_rec_index = i

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}_{}.pth'.format(k, i))
                if not os.path.exists(save_file):
                    torch.save(model.state_dict(), save_file)

        if dev_f1 < dev_f1_:
            dev_f1 = dev_f1_
            max_f1_index = i

            if args.save_model:
                save_file = os.path.join(args.output_path, 'model_{}_{}.pth'.format(k, i))
                if not os.path.exists(save_file):
                    torch.save(model.state_dict(), save_file)

        if i - max_f1_index > args.stop_num:
            break

    file = open('result.txt', 'a')
    file.write('max_pre: {}_{}, {}'.format(k, max_pre_index, dev_pre) + '\n')
    file.write('max_rec: {}_{}, {}'.format(k, max_rec_index, dev_rec) + '\n')
    file.write('max_f1: {}_{}, {}'.format(k, max_f1_index, dev_f1) + '\n' + '\n')
    file.close()

    print('-----------------------------------------------------------------------------------------------------------')
    print('max_pre: {}_{}, {}'.format(k, max_pre_index, dev_pre))
    print('max_rec: {}_{}, {}'.format(k, max_rec_index, dev_rec))
    print('max_f1: {}_{}, {}'.format(k, max_f1_index, dev_f1))
    print('-----------------------------------------------------------------------------------------------------------')


def main():
    args = init_arg_parser()
    setup_seed(args=args)

    if args.trail_train:
        model = bert_classifi(args=args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        fc_para = list(map(id, model.fc.parameters()))
        base_para = filter(lambda p: id(p) not in fc_para, model.parameters())
        params = [{'params': base_para},
                  {'params': model.fc.parameters(), 'lr': args.fc_lr}]

        optimizer = Adam(params, lr=args.bert_lr, amsgrad=args.use_amsgrad)

        if args.use_label_smoothing:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        train_trail(model, optimizer, criterion, args)
    else:
        cross_train(args)


if __name__ == '__main__':
    main()


