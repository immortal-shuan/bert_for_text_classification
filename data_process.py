import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


def data_process(args):
    data = data_read(args=args)
    if args.is_train:
        data = convert_label(data, args.label_info)
    train_ids = convert_text_to_id(data, args)
    if args.trail_train:
        train_ids, dev_ids, dev_label_num = train_dev_split(train_ids, args)
        if args.use_label_smoothing:
            train_ids = convert_smooth_label(train_ids, args)
            dev_ids = convert_smooth_label(dev_ids, args)
        return train_ids, dev_ids, dev_label_num
    if args.use_label_smoothing:
        train_ids = convert_smooth_label(train_ids, args)
    return train_ids


def data_read(args):
    data = []
    if args.is_train:
        with open(args.train_path, 'r', encoding='utf-8-sig') as df:
            for line in df:
                sample = line.strip().split('\t')
                assert len(sample) == 3
                data.append([int(sample[0]), sample[1], sample[2]])
            df.close()
        return data
    else:
        with open(args.test_path, 'r', encoding='utf-8-sig') as df:
            for line in df:
                sample = line.strip().split('\t')
                assert len(sample) == 2
                data.append([int(sample[0]), sample[1]])
            df.close()
        return data


def convert_label(data, label_dict):
    for i in range(len(data)):
        label = data[i][-1]
        assert label in label_dict.keys()
        data[i][-1] = label_dict[label]
    return data


def convert_smooth_label(data, args):
    new_data = []
    label_smooth = args.label_smooth
    for sample in tqdm(data):
        label = sample[-1]
        new_label = [0.0] * args.class_num
        new_label[label] = 1.0
        smooth_label = (1.0-label_smooth)*np.array(new_label) + label_smooth/args.class_num
        new_data.append([sample[0], sample[1], sample[2], smooth_label.tolist()])
    return new_data


def convert_text_to_id(data, args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    new_data = []
    for i in trange(len(data)):

        if args.is_train:
            assert len(data[i]) == 3
            input_info = tokenizer.encode_plus(data[i][1], add_special_tokens=True)
            new_data.append([input_info['input_ids'], input_info['token_type_ids'], input_info['attention_mask'],
                             data[i][2]])
        else:
            assert len(data[i]) == 2
            input_info = tokenizer.encode_plus(data[i][1], add_special_tokens=True)
            new_data.append([input_info['input_ids'], input_info['token_type_ids'], input_info['attention_mask']])
    return new_data


def train_dev_split(data, args):
    new_data = {i: [] for i in range(args.class_num)}
    for sample in tqdm(data):
        label = sample[-1]
        new_data[label].append(sample)
    train_ids = []
    dev_ids = []
    dev_label_num = []
    for label in new_data.keys():
        sub_train_ids, sub_dev_ids = train_test_split(
            new_data[label], train_size=args.train_size, random_state=args.seed
        )
        train_ids.extend(sub_train_ids)
        dev_ids.extend(sub_dev_ids)
        dev_label_num.append(len(sub_dev_ids))
    return train_ids, dev_ids, dev_label_num


def Batch(train_data, args):
    text_input = []
    text_type = []
    text_mask = []
    if args.is_train:
        label = []
    for i in range(len(train_data)):
        text_input.append(train_data[i][0])
        text_type.append(train_data[i][1])
        text_mask.append(train_data[i][2])

        if args.is_train:
            label.append(train_data[i][3])

    text_input_ = batch_pad(text_input, args, pad=0)
    text_type_ = batch_pad(text_type, args, pad=0)
    text_mask_ = batch_pad(text_mask, args, pad=0)

    if args.is_train:
        if args.use_label_smoothing:
            return text_input_, text_type_, text_mask_, torch.tensor(label, dtype=torch.float).cuda()
        else:
            return text_input_, text_type_, text_mask_, torch.tensor(label, dtype=torch.long).cuda()
    else:
        return text_input_, text_type_, text_mask_


def batch_pad(batch_data, args, pad=0):
    seq_len = [len(i) for i in batch_data]
    max_len = max(seq_len)
    if max_len > args.max_len:
        max_len = args.max_len
    out = []
    for line in batch_data:
        if len(line) < max_len:
            out.append(line + [pad] * (max_len - len(line)))
        else:
            out.append(line[:args.max_len])
    return torch.tensor(out, dtype=torch.long).cuda()


if __name__ == '__main__':
    from arg import init_arg_parser
    args = init_arg_parser()

    train_ids, dev_ids, dev_label_num = data_process(args)

