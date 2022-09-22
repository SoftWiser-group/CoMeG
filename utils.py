import logging
import pickle

import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_examples(file_name):
    """
    format: [[pr,cid,date,msg,tok1,sub1,att1,tok2,sub2,att2,idx1,idx2,act]]
    """
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def convert_examples_to_features(examples, src_vocab, node_vocab, tgt_vocab, diff_vocab, args, stage=None):
    """
    examples: [([pr,cid,date,msg,tok1,sub1,att1,tok2,sub2,att2,idx1,idx2,act],diff)]
    return:     diff_ids=None,
                sub_token_ids1=None,
                ast_att_ids1=None,
                mask1=None,
                sub_token_ids2=None,
                ast_att_ids2=None,
                mask2=None,
                chg_idxs1=None,
                chg_mask1=None,
                chg_idxs2=None,
                chg_mask2=None,
                chg_type=None,
                copy_ids1=None,
                copy_ids2=None,
                tgt_ids=None,
                tgt_mask=None,
    """
    features = []
    type2id = {'INSERT': 1, 'DELETE': 2, 'MOVE': 3, 'UPDATE': 4}
    for ex_idx, (example, diff) in enumerate(examples):
        # 处理diff
        diff_ids = np.zeros(args.max_diff, dtype=np.int32)
        diff_ids[0] = 3 # unk_id, 防止出现全0
        for i in range(min(args.max_diff, len(diff))):
            diff_ids[i] = diff_vocab[diff[i]]

        copy_ids_dict = dict()
        assert len(example[5]) == len(example[6])
        # 因为pad id为0，所以直接使用np.zeros，不通用
        sub_token_ids1 = np.zeros((args.max_src, args.max_sub), dtype=np.int32)
        ast_att_ids1 = np.zeros((args.max_src, args.max_att), dtype=np.int32)
        mask1 = np.zeros(args.max_src, dtype=np.bool8)
        copy_ids1 = np.zeros(args.max_src, dtype=np.int32)
        # 5是sub1 6是att1
        for i in range(min(args.max_src, len(example[5]))):
            mask1[i] = True
            sub = example[5][i]
            # 计算token在tgt中的id用于拷贝
            token = ''.join(sub)
            token_tgt_id = tgt_vocab[token]
            # id为unk_id时重新赋一个与位置有关的id
            if token_tgt_id == tgt_vocab.unk_id:
                if token not in copy_ids_dict:
                    copy_ids_dict[token] = len(tgt_vocab) + i
                token_tgt_id = copy_ids_dict[token]
            copy_ids1[i] = token_tgt_id
            for j in range(min(args.max_sub, len(sub))):
                sub_token_ids1[i, j] = src_vocab[sub[j]]
            att = example[6][i]
            for j in range(min(args.max_att, len(att))):
                ast_att_ids1[i, j] = node_vocab[att[j]]

        assert len(example[8]) == len(example[9])
        sub_token_ids2 = np.zeros((args.max_src, args.max_sub), dtype=np.int32)
        ast_att_ids2 = np.zeros((args.max_src, args.max_att), dtype=np.int32)
        mask2 = np.zeros(args.max_src, dtype=np.bool8)
        copy_ids2 = np.zeros(args.max_src, dtype=np.int32)
        for i in range(min(args.max_src, len(example[8]))):
            mask2[i] = True
            sub = example[8][i]
            # 计算token在tgt中的id用于拷贝
            token = ''.join(sub)
            token_tgt_id = tgt_vocab[token]
            # id为unk_id时重新赋一个与位置有关的id
            if token_tgt_id == tgt_vocab.unk_id:
                if token not in copy_ids_dict:
                    copy_ids_dict[token] = len(tgt_vocab) + i + args.max_src
                token_tgt_id = copy_ids_dict[token]
            copy_ids2[i] = token_tgt_id
            for j in range(min(args.max_sub, len(sub))):
                sub_token_ids2[i, j] = src_vocab[sub[j]]
            att = example[9][i]
            for j in range(min(args.max_att, len(att))):
                ast_att_ids2[i, j] = node_vocab[att[j]]

        assert len(example[10]) == len(example[11]) == len(example[12])
        chg_idxs1 = np.zeros(args.max_chg, dtype=np.int32)
        chg_mask1 = np.zeros(args.max_chg, dtype=np.bool8)
        chg_idxs2 = np.zeros(args.max_chg, dtype=np.int32)
        chg_mask2 = np.zeros(args.max_chg, dtype=np.bool8)
        chg_type = np.zeros(args.max_chg, dtype=np.int32)
        for i in range(min(args.max_chg, len(example[10]))):
            idx1 = example[10][i]
            idx2 = example[11][i]
            type = type2id.get(example[12][i], 0)
            if 0 <= idx1 < args.max_src:
                chg_idxs1[i] = idx1
                chg_mask1[i] = True
            if 0 <= idx2 < args.max_src:
                chg_idxs2[i] = idx2
                chg_mask2[i] = True
            chg_type[i] = type

        # 处理target
        if stage == 'test':
            tgt_tokens = []
        else:
            tgt_tokens = example[3]
        tgt_tokens = ['<s>'] + tgt_tokens + ['</s>']
        tgt_ids = np.zeros(args.max_tgt, dtype=np.int32)
        tgt_mask = np.zeros(args.max_tgt, dtype=np.bool8)
        for i in range(min(args.max_tgt, len(tgt_tokens))):
            id = tgt_vocab[tgt_tokens[i]]
            # id为unk_id时，看是不是可拷贝id
            if id == tgt_vocab.unk_id:
                id = copy_ids_dict.get(tgt_tokens[i], tgt_vocab.unk_id)
            tgt_ids[i] = id
            tgt_mask[i] = True

        if ex_idx < 2:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info('diff_ids')
                logger.info(diff_ids)
                logger.info('sub_token_ids1')
                logger.info(sub_token_ids1)
                logger.info('ast_att_ids1')
                logger.info(ast_att_ids1)
                logger.info('mask1')
                logger.info(mask1)
                logger.info('sub_token_ids2')
                logger.info(sub_token_ids2)
                logger.info('ast_att_ids2')
                logger.info(ast_att_ids2)
                logger.info('mask2')
                logger.info(mask2)
                logger.info('chg_idxs1')
                logger.info(chg_idxs1)
                logger.info('chg_mask1')
                logger.info(chg_mask1)
                logger.info('chg_idxs2')
                logger.info(chg_idxs2)
                logger.info('chg_mask2')
                logger.info(chg_mask2)
                logger.info('chg_type')
                logger.info(chg_type)
                logger.info('copy_ids1')
                logger.info(copy_ids1)
                logger.info('copy_ids2')
                logger.info(copy_ids2)
                logger.info('tgt_ids')
                logger.info(tgt_ids)
                logger.info('tgt_mask')
                logger.info(tgt_mask)
        features.append(dotdict(
            diff_ids=diff_ids,
            sub_token_ids1=sub_token_ids1,
            ast_att_ids1=ast_att_ids1,
            mask1=mask1,
            sub_token_ids2=sub_token_ids2,
            ast_att_ids2=ast_att_ids2,
            mask2=mask2,
            chg_idxs1=chg_idxs1,
            chg_mask1=chg_mask1,
            chg_idxs2=chg_idxs2,
            chg_mask2=chg_mask2,
            chg_type=chg_type,
            copy_ids1=copy_ids1,
            copy_ids2=copy_ids2,
            tgt_ids=tgt_ids,
            tgt_mask=tgt_mask,
        ))

    return features


def main():
    # TODO: need updated
    from vocab import VocabEntry, read_diff_from_jsonl
    examples = read_examples('/data/share/kingxu/data/MyData/by_pr/test.pkl')
    diff = read_diff_from_jsonl('/data/share/kingxu/data/MyData/nmt_nngen/by_pr/test.jsonl')
    examples = list(zip(examples, diff))
    src_vocab = VocabEntry.load('/data/share/kingxu/data/MyData/src_vocab.json')
    node_vocab = VocabEntry.load('/data/share/kingxu/data/MyData/node_vocab.json')
    tgt_vocab = VocabEntry.load('/data/share/kingxu/data/MyData/tgt_vocab.json')
    diff_vocab = VocabEntry.load('/data/share/kingxu/data/MyData/diff_vocab.json')
    args = dotdict(
        max_src=250,
        max_sub=8,
        max_att=8,
        max_tgt=20,
        max_chg=50,
        max_diff=100
    )
    features = convert_examples_to_features(examples, src_vocab, node_vocab, tgt_vocab, diff_vocab, args, 'train')


if __name__ == '__main__':
    main()
