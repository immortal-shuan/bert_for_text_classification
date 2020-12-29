import argparse


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    # 数据预处理时的参数
    arg_parser.add_argument('--train_size', default=0.8)             # 训练集所占比例
    arg_parser.add_argument('--max_len', default=512)                # 句子的最大长度
    arg_parser.add_argument('--use_label_smoothing', default=True)  # 是否使用软标签
    arg_parser.add_argument('--label_smooth', default=0.1)           # 软标签阈值
    arg_parser.add_argument('--add_key_words', default=False)        # 是否添加关键词
    arg_parser.add_argument('--key_words_num', default=2)            # 添加关键词的数目
    arg_parser.add_argument('--max_df', default=1.0)                 # tfidf中的词在句子中出现的次数占据所有句子的最大比
    arg_parser.add_argument('--min_df', default=1)                   # tfidf中的词在句子中出现的最少次数
    arg_parser.add_argument('--rm_stopwords', default=True)          # tfidf中的词在句子中出现的最少次数
    arg_parser.add_argument('--label_info', default={
        '100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '106': 5, '107': 6, '108': 7, '109': 8, '110': 9, '112': 10,
        '113': 11, '114': 12, '115': 13, '116': 14})                 # tfidf中的词在句子中出现的最少次数

    # 训练时参数
    arg_parser.add_argument('--stop_num', default=4)         # 当评价指标在stop_num内不在增长时，训练停止
    arg_parser.add_argument('--trail_train', default=True)   # 当评价指标在stop_num内不在增长时，训练停止
    arg_parser.add_argument('--seed', default=102)           # 随机种子
    arg_parser.add_argument('--epoch_num', default=20)          # 数据训练多少轮
    arg_parser.add_argument('--batch_size', default=32)      # 模型一次输入的数据
    arg_parser.add_argument('--save_model', default=False)   # 是否保存模型
    arg_parser.add_argument('--use_amsgrad', default=False)  # 是否使用amsgrad
    arg_parser.add_argument('--is_train', default=True)      # 是否是训练
    arg_parser.add_argument('--use_kfold', default=True)     # 是否使用交叉验证
    arg_parser.add_argument('--k', default=10)               # 采用几折交叉验证
    arg_parser.add_argument('--use_fgm', default=True)       # 是否使用fgm对抗训练
    arg_parser.add_argument('--use_pdg', default=False)      # 是否使用pdg对抗训练
    arg_parser.add_argument('--k_pdg', default=3)            # pdg对抗训练的次数
    arg_parser.add_argument('--loss_step', default=2)        # 模型几次loss后更新参数
    arg_parser.add_argument('--out_dev_result', default=False)  # 是否输出评估集每个标签的得分情况

    # 各种文档路径
    arg_parser.add_argument('--train_path', default='TNEWS/TNEWS_train1128.csv')     # 训练集的文件路径
    arg_parser.add_argument('--test_path', default='TNEWS/TNEWS_a.csv')              # 测试集的文件路径
    arg_parser.add_argument('--bert_path', default='roberta_large')                 # 预训练模型的文件路径
    arg_parser.add_argument('--output_path', default='model_output')                # 模型输出路径
    arg_parser.add_argument('--stopwords_path', default='stopwords/hagongda_stopwords.txt')     # 停用词路径

    # 模型内各种参数
    arg_parser.add_argument('--fc_lr', default=2e-3)            # 分类层学习率
    arg_parser.add_argument('--other_lr', default=2e-3)         # 其他层学习率
    arg_parser.add_argument('--bert_lr', default=2e-5)          # bert层学习率
    arg_parser.add_argument('--fc_dropout', default=0.5)        # 分类层dropout
    arg_parser.add_argument('--use_bert_dropout', default=False)  # bert输出层dropout
    arg_parser.add_argument('--bert_dropout', default=0.15)     # bert输出层dropout
    arg_parser.add_argument('--bert_dim', default=1024)         # bert的输出向量维度
    arg_parser.add_argument('--lstm_hidden_dim', default=512)   # lstm的隐藏层
    arg_parser.add_argument('--bilstm', default=True)           # lstm是否双向
    arg_parser.add_argument('--output_hidden_states', default=True)  # 是否输出隐藏层
    arg_parser.add_argument('--use_cls', default=False)         # 是否使用cls
    arg_parser.add_argument('--class_num', default=15)          # 输出的维度

    args = arg_parser.parse_args()
    return args


