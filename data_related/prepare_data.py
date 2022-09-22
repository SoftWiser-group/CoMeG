from collections import defaultdict
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Counter

from tqdm import tqdm

from load_gumtree_results import process_tree_diff
from load_messages import process_msg

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data():
    res = []
    # We do not provide the original commits.jsonl file
    with open('/data/share/kingxu/data/MyData/commits.jsonl') as f:
        for l in tqdm(f):
            obj = json.loads(l)
            msg = process_msg(obj)
            if len(msg) >= 20 or len(msg) < 3:
                continue
            diff = process_tree_diff(obj)
            if len(diff[0]) == 0 or len(diff[0]) >= 250 or len(diff[3]) == 0 or len(diff[3]) >= 250 or len(diff[6]) >= 50 or len(diff[6]) == 0:
                continue
            res.append((obj['pr'], obj['cid'], obj['date'], msg,
                        diff[0], diff[1], diff[2], diff[3], diff[4], diff[5], diff[6], diff[7], diff[8]))
    logger.info(f'total {len(res)} records!')
    logger.info('Saving as pkl ...')
    with open("/data/share/kingxu/data/MyData/pr_cid_date_msg_tok1_sub1_att1_tok2_sub2_att2_idx1_idx2_act.pkl", 'wb') as p:
        pickle.dump(res, p)
    logger.info('Done!')


def load_data():
    """
    format: [[pr,cid,date,msg,tok1,sub1,att1,tok2,sub2,att2,idx1,idx2,act]]
    """
    with open("/data/share/kingxu/data/MyData/pr_cid_date_msg_tok1_sub1_att1_tok2_sub2_att2_idx1_idx2_act.pkl", 'rb') as p:
        data = pickle.load(p)
    return data


def split_data_by_time(data):
    length = len(data)
    t_p = length // 10
    logger.info(f"10%: {t_p}")
    data.sort(key=lambda x: int(x[2]))
    save_dir = Path("/data/share/kingxu/data/MyData") / 'by_time'
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    with open(save_dir / 'train.pkl', 'wb') as f:
        pickle.dump(data[:length-t_p-t_p], f)
    with open(save_dir / 'valid.pkl', 'wb') as f:
        pickle.dump(data[length-t_p-t_p:length-t_p], f)
    with open(save_dir / 'test.pkl', 'wb') as f:
        pickle.dump(data[length-t_p:], f)


def split_data_by_pr(data):
    length = len(data)
    t_p = length // 10
    pr_num = defaultdict(int)
    for i in data:
        pr_num[i[0]] += 1
    pr_len = len(pr_num)
    t_p_pr = pr_len // 10
    while True:
        val_test_pr = random.sample(list(pr_num.keys()), t_p_pr + t_p_pr)
        val_count, test_count = 0, 0
        val_pr, test_pr = [], []
        for i, p in enumerate(val_test_pr):
            if i % 2 == 0:
                val_count += pr_num[p]
                val_pr.append(p)
            else:
                test_count += pr_num[p]
                test_pr.append(p)
        if 0 < t_p - val_count < 1000 and 0 < t_p - test_count < 1000:
            break
    logger.info(f"valid: {val_count} {val_pr}")
    logger.info(f'test:  {test_count} {test_pr}')
    train, valid, test = [], [], []
    for i in data:
        if i[0] in val_pr:
            valid.append(i)
        elif i[0] in test_pr:
            test.append(i)
        else:
            train.append(i)
    save_dir = Path("/data/share/kingxu/data/MyData") / 'by_pr'
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    with open(save_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(save_dir / 'valid.pkl', 'wb') as f:
        pickle.dump(valid, f)
    with open(save_dir / 'test.pkl', 'wb') as f:
        pickle.dump(test, f)


def split_data_randomly(data):
    length = len(data)
    t_p = length // 10
    logger.info(f"10%: {t_p}")
    random.shuffle(data)
    save_dir = Path("/data/share/kingxu/data/MyData") / 'randomly'
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    with open(save_dir / 'train.pkl', 'wb') as f:
        pickle.dump(data[:length-t_p-t_p], f)
    with open(save_dir / 'valid.pkl', 'wb') as f:
        pickle.dump(data[length-t_p-t_p:length-t_p], f)
    with open(save_dir / 'test.pkl', 'wb') as f:
        pickle.dump(data[length-t_p:], f)


def split_data():
    data = load_data()
    logger.info('split data randomly !')
    split_data_randomly(data)
    logger.info('split data by time !')
    split_data_by_time(data)
    logger.info('split data by pr !')
    split_data_by_pr(data)


if __name__ == '__main__':
    # The file shows the procedures that we process our dataset, and we provide the processed file.
    # prepare_data()
    split_data()
