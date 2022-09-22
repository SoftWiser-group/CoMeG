from __future__ import absolute_import

import argparse
import json
import logging
import os
from pathlib import Path
import random
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

import bleu
from eval_utils import compute_bleu, compute_meteor_rouge
from model import AstDiff2Seq
from utils import convert_examples_to_features, dotdict, read_examples
from vocab import VocabEntry, read_diff_from_jsonl

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    # Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--train_diff_file", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_diff_file", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_diff_file", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    # Vocab
    parser.add_argument("--src_vocab_file", default="", type=str,
                        help="Saved vocab file.")
    parser.add_argument("--tgt_vocab_file", default="", type=str,
                        help="taget vocab file.")
    parser.add_argument("--node_vocab_file", default="", type=str,
                        help="node vocab file.")
    parser.add_argument("--diff_vocab_file", default="/data/share/kingxu/data/MyData/diff_vocab.json", type=str,
                        help="diff vocab file.")
    # Dataset parameters
    parser.add_argument("--max_src", default=250, type=int,
                        help="The maximum total word length. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_tgt", default=20, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_chg", default=50, type=int,
                        help="The maximum total change sequence length after encode code. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_sub", default=8, type=int,
                        help="the num of subwords in one word. do truncated and padded.")
    parser.add_argument("--max_att", default=8, type=int,
                        help="The num of nodes in one path. do truncated and padded")
    parser.add_argument("--max_diff", default=100, type=int,
                        help="The num of diff tokens. do truncated and padded")

    # 模型参数,用于对config参数进行修改
    parser.add_argument("--num_enc_layers", default=3, type=int,
                        help="The num of subword encoder layers")
    parser.add_argument("--num_chg_layers", default=1, type=int,
                        help="The num of change encoder layers")
    parser.add_argument("--num_dec_layers", default=3, type=int,
                        help="The num of decoder layers")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="The size of hidden state vectors")
    parser.add_argument("--num_attention_heads", default=8, type=int,
                        help="The num of attention heads")
    parser.add_argument("--intermediate_size", default=1024, type=int,
                        help="The dimension of the feedforward network model")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # make dir if output_dir not exist
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # Save args into file
    if args.do_train:
        with open(output_dir / 'train_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)
    else:
        with open(output_dir / 'test_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)

    # Load and set config and vocabulary
    # config = RobertaConfig.from_pretrained(args.config_name)
    # vocab = VocabEntry.load(args.vocab_file)
    # 选出一些可以配置的选项
    # config.num_enc_layers = args.num_enc_layers
    # config.num_word_layers = args.num_word_layers
    # config.num_chg_layers = args.num_chg_layers
    # config.num_dec_layers = args.num_dec_layers

    # config.node_embed_size = 64  # 暂时不准备更改
    # config.num_attention_heads = args.num_attention_heads
    # config.hidden_size = args.hidden_size
    # config.intermediate_size = args.intermediate_size

    # config.num_subwords = args.num_subwords
    # config.num_nodes = args.num_nodes

    # Build vocab
    src_vocab = VocabEntry.load(args.src_vocab_file)
    tgt_vocab = VocabEntry.load(args.tgt_vocab_file)
    node_vocab = VocabEntry.load(args.node_vocab_file)
    diff_vocab = VocabEntry.load(args.diff_vocab_file)

    # Build config
    config1 = dotdict(vocab_size=len(src_vocab),
                      hid_size=args.hidden_size,
                      pad_id=0,
                      layers=1,
                      dropout=0.1,)
    config2 = dotdict(vocab_size=len(node_vocab),
                      hid_size=args.hidden_size,
                      pad_id=0,
                      layers=1,
                      dropout=0.1,)
    config3 = dotdict(embedder_config=config1,
                      node_embedder_config=config2,
                      hid_size=args.hidden_size,
                      vocab_size=len(tgt_vocab),
                      diff_vocab_size=len(diff_vocab),
                      attn_heads=args.num_attention_heads,
                      intermediate_size=args.intermediate_size,
                      enc_layers=args.num_enc_layers,
                      chg_layers=1,
                      dec_layers=args.num_dec_layers,
                      max_pos=args.max_src,
                      layer_norm_eps=1e-6,
                      sos_id=1,
                      eos_id=2,
                      pad_id=0,
                      chg_vocab=6,
                      dropout=0.1,)
    # Build model
    model = AstDiff2Seq(config=config3, beam_size=args.beam_size, max_length=args.max_tgt)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        msgs = model.load_state_dict(torch.load(args.load_model_path), strict=False)
        logger.info(f"Reload status: {msgs}")

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_diff = read_diff_from_jsonl(args.train_diff_file)
        assert len(train_examples) == len(train_diff)
        train_examples = list(zip(train_examples, train_diff))
        train_features = convert_examples_to_features(train_examples, src_vocab, node_vocab, tgt_vocab, diff_vocab, args, stage='train')
        """ 
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
        """
        all0 = torch.tensor([f.diff_ids for f in train_features], dtype=torch.long)
        all1 = torch.tensor([f.sub_token_ids1 for f in train_features], dtype=torch.long)
        all2 = torch.tensor([f.ast_att_ids1 for f in train_features], dtype=torch.long)
        all3 = torch.tensor([f.mask1 for f in train_features], dtype=torch.bool)
        all4 = torch.tensor([f.sub_token_ids2 for f in train_features], dtype=torch.long)
        all5 = torch.tensor([f.ast_att_ids2 for f in train_features], dtype=torch.long)
        all6 = torch.tensor([f.mask2 for f in train_features], dtype=torch.bool)
        all7 = torch.tensor([f.chg_idxs1 for f in train_features], dtype=torch.long)
        all8 = torch.tensor([f.chg_mask1 for f in train_features], dtype=torch.bool)
        all9 = torch.tensor([f.chg_idxs2 for f in train_features], dtype=torch.long)
        all10 = torch.tensor([f.chg_mask2 for f in train_features], dtype=torch.bool)
        all11 = torch.tensor([f.chg_type for f in train_features], dtype=torch.long)
        all12 = torch.tensor([f.copy_ids1 for f in train_features], dtype=torch.long)
        all13 = torch.tensor([f.copy_ids2 for f in train_features], dtype=torch.long)
        all14 = torch.tensor([f.tgt_ids for f in train_features], dtype=torch.long)
        all15 = torch.tensor([f.tgt_mask for f in train_features], dtype=torch.bool)
        train_data = TensorDataset(
            all0,
            all1,
            all2,
            all3,
            all4,
            all5,
            all6,
            all7,
            all8,
            all9,
            all10,
            all11,
            all12,
            all13,
            all14,
            all15,
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        loss_patience, bleu_patience = 0, 0
        # bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        # train_dataloader = cycle(train_dataloader) # 使用cycle，每个epoch之后不会对数据重新shuffle
        # eval_flag = True
        for epoch in range(1, args.num_train_epochs + 1):
            logger.info(f" Epoch: {epoch}")
            bar = tqdm(train_dataloader)
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                loss, _, _ = model(*batch)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps  # 先除再加等于加完再除
                # 用于显示
                tr_loss += loss.item()
                train_loss = round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1), 4)
                bar.set_description("loss {}".format(train_loss))
                nb_tr_examples += batch[0].size(0)
                nb_tr_steps += 1

                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            # 因为optimizer是有条件更新，循环结束时不一定更新参数，补上
            optimizer.step()
            optimizer.zero_grad()

            if args.do_eval:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_diff = read_diff_from_jsonl(args.dev_diff_file)
                    assert len(eval_examples) == len(eval_diff)
                    eval_examples = list(zip(eval_examples, eval_diff))
                    eval_features = convert_examples_to_features(eval_examples, src_vocab, node_vocab, tgt_vocab, diff_vocab, args, stage='dev')

                    all0 = torch.tensor([f.diff_ids for f in eval_features], dtype=torch.long)
                    all1 = torch.tensor([f.sub_token_ids1 for f in eval_features], dtype=torch.long)
                    all2 = torch.tensor([f.ast_att_ids1 for f in eval_features], dtype=torch.long)
                    all3 = torch.tensor([f.mask1 for f in eval_features], dtype=torch.bool)
                    all4 = torch.tensor([f.sub_token_ids2 for f in eval_features], dtype=torch.long)
                    all5 = torch.tensor([f.ast_att_ids2 for f in eval_features], dtype=torch.long)
                    all6 = torch.tensor([f.mask2 for f in eval_features], dtype=torch.bool)
                    all7 = torch.tensor([f.chg_idxs1 for f in eval_features], dtype=torch.long)
                    all8 = torch.tensor([f.chg_mask1 for f in eval_features], dtype=torch.bool)
                    all9 = torch.tensor([f.chg_idxs2 for f in eval_features], dtype=torch.long)
                    all10 = torch.tensor([f.chg_mask2 for f in eval_features], dtype=torch.bool)
                    all11 = torch.tensor([f.chg_type for f in eval_features], dtype=torch.long)
                    all12 = torch.tensor([f.copy_ids1 for f in eval_features], dtype=torch.long)
                    all13 = torch.tensor([f.copy_ids2 for f in eval_features], dtype=torch.long)
                    all14 = torch.tensor([f.tgt_ids for f in eval_features], dtype=torch.long)
                    all15 = torch.tensor([f.tgt_mask for f in eval_features], dtype=torch.bool)

                    eval_data = TensorDataset(
                        all0,
                        all1,
                        all2,
                        all3,
                        all4,
                        all5,
                        all6,
                        all7,
                        all8,
                        all9,
                        all10,
                        all11,
                        all12,
                        all13,
                        all14,
                        all15,
                    )
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    with torch.no_grad():
                        _, loss, num = model(*batch)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)

                # save last checkpoint
                last_output_dir = output_dir / 'checkpoint-last'
                if not last_output_dir.exists():
                    last_output_dir.mkdir(parents=True)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = last_output_dir / "pytorch_model.bin"
                torch.save(model_to_save.state_dict(), output_model_file)
                loss_patience += 1
                if eval_loss < best_loss:
                    loss_patience = 0
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  "+"*"*20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    bl_output_dir = output_dir / 'checkpoint-best-ppl'
                    if not bl_output_dir.exists():
                        bl_output_dir.mkdir(parents=True)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = bl_output_dir / "pytorch_model.bin"
                    torch.save(model_to_save.state_dict(), output_model_file)

                # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_diff = read_diff_from_jsonl(args.dev_diff_file)
                    # assert len(eval_examples) == len(eval_diff)
                    eval_examples = list(zip(eval_examples, eval_diff))
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, src_vocab, node_vocab, tgt_vocab, diff_vocab, args, stage='test')
                    all0 = torch.tensor([f.diff_ids for f in eval_features], dtype=torch.long)
                    all1 = torch.tensor([f.sub_token_ids1 for f in eval_features], dtype=torch.long)
                    all2 = torch.tensor([f.ast_att_ids1 for f in eval_features], dtype=torch.long)
                    all3 = torch.tensor([f.mask1 for f in eval_features], dtype=torch.bool)
                    all4 = torch.tensor([f.sub_token_ids2 for f in eval_features], dtype=torch.long)
                    all5 = torch.tensor([f.ast_att_ids2 for f in eval_features], dtype=torch.long)
                    all6 = torch.tensor([f.mask2 for f in eval_features], dtype=torch.bool)
                    all7 = torch.tensor([f.chg_idxs1 for f in eval_features], dtype=torch.long)
                    all8 = torch.tensor([f.chg_mask1 for f in eval_features], dtype=torch.bool)
                    all9 = torch.tensor([f.chg_idxs2 for f in eval_features], dtype=torch.long)
                    all10 = torch.tensor([f.chg_mask2 for f in eval_features], dtype=torch.bool)
                    all11 = torch.tensor([f.chg_type for f in eval_features], dtype=torch.long)
                    all12 = torch.tensor([f.copy_ids1 for f in eval_features], dtype=torch.long)
                    all13 = torch.tensor([f.copy_ids2 for f in eval_features], dtype=torch.long)
                    # all14 = torch.tensor([f.tgt_ids for f in eval_features], dtype=torch.long)
                    # all15 = torch.tensor([f.tgt_mask for f in eval_features], dtype=torch.bool)

                    eval_data = TensorDataset(
                        all0,
                        all1,
                        all2,
                        all3,
                        all4,
                        all5,
                        all6,
                        all7,
                        all8,
                        all9,
                        all10,
                        all11,
                        all12,
                        all13,
                        # all14,
                        # all15,
                    )

                    dev_dataset['dev_bleu'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    with torch.no_grad():
                        preds = model(*batch)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            p.append(t)
                model.train()
                predictions = []
                with open(output_dir / "dev.output", 'w') as f, open(output_dir / "dev.gold", 'w') as f1:
                    for p_ids, (gold, _) in zip(p, eval_examples):
                        p_tokens = tgt_vocab.decode(p_ids, gold, args)
                        predictions.append(f'{gold[0]}{gold[1]}'+'\t'+' '.join(p_tokens))
                        f.write(f'{gold[0]}{gold[1]}'+'\t'+' '.join(p_tokens)+'\n')
                        f1.write(f'{gold[0]}{gold[1]}'+'\t'+' '.join(gold[3])+'\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, output_dir / "dev.gold")
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  "+"*"*20)
                bleu_patience += 1
                if dev_bleu > best_bleu:
                    bleu_patience = 0
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    bb_output_dir = output_dir / 'checkpoint-best-bleu'
                    if not bb_output_dir.exists():
                        bb_output_dir.mkdir(parents=True)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = bb_output_dir / "pytorch_model.bin"
                    torch.save(model_to_save.state_dict(), output_model_file)

                if loss_patience >= 3 and bleu_patience >= 3:
                    logger.info("*** Early Stop ***")
                    exit(0)

    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append((args.dev_filename, args.dev_diff_file))
        if args.test_filename is not None:
            files.append((args.test_filename, args.test_diff_file))
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file[0]))
            eval_examples = read_examples(file[0])
            eval_diff = read_diff_from_jsonl(file[1])
            assert len(eval_examples) == len(eval_diff)
            eval_examples = list(zip(eval_examples, eval_diff))
            eval_features = convert_examples_to_features(eval_examples, src_vocab, node_vocab, tgt_vocab, diff_vocab, args, stage='test')

            all0 = torch.tensor([f.diff_ids for f in eval_features], dtype=torch.long)
            all1 = torch.tensor([f.sub_token_ids1 for f in eval_features], dtype=torch.long)
            all2 = torch.tensor([f.ast_att_ids1 for f in eval_features], dtype=torch.long)
            all3 = torch.tensor([f.mask1 for f in eval_features], dtype=torch.bool)
            all4 = torch.tensor([f.sub_token_ids2 for f in eval_features], dtype=torch.long)
            all5 = torch.tensor([f.ast_att_ids2 for f in eval_features], dtype=torch.long)
            all6 = torch.tensor([f.mask2 for f in eval_features], dtype=torch.bool)
            all7 = torch.tensor([f.chg_idxs1 for f in eval_features], dtype=torch.long)
            all8 = torch.tensor([f.chg_mask1 for f in eval_features], dtype=torch.bool)
            all9 = torch.tensor([f.chg_idxs2 for f in eval_features], dtype=torch.long)
            all10 = torch.tensor([f.chg_mask2 for f in eval_features], dtype=torch.bool)
            all11 = torch.tensor([f.chg_type for f in eval_features], dtype=torch.long)
            all12 = torch.tensor([f.copy_ids1 for f in eval_features], dtype=torch.long)
            all13 = torch.tensor([f.copy_ids2 for f in eval_features], dtype=torch.long)
            # all14 = torch.tensor([f.tgt_ids for f in eval_features], dtype=torch.long)
            # all15 = torch.tensor([f.tgt_mask for f in eval_features], dtype=torch.bool)

            eval_data = TensorDataset(
                all0,
                all1,
                all2,
                all3,
                all4,
                all5,
                all6,
                all7,
                all8,
                all9,
                all10,
                all11,
                all12,
                all13,
                # all14,
                # all15,
            )

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                with torch.no_grad():
                    preds = model(*batch)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        p.append(t)
            model.train()
            predictions = []
            references = []
            with open(output_dir / f"test_{idx}.output", 'w') as f, open(output_dir / f"test_{idx}.gold", 'w') as f1:
                for p_ids, (gold, _) in zip(p, eval_examples):
                    p_tokens = tgt_vocab.decode(p_ids, gold, args)
                    f.write(f'{gold[0]}{gold[1]}'+'\t'+" ".join(p_tokens)+'\n')
                    gold_str = ' '.join(gold[3])
                    f1.write(f'{gold[0]}{gold[1]}'+'\t'+gold_str+'\n')
                    predictions.append(p_tokens)
                    references.append([gold[3]])

            # (goldMap, predictionMap) = bleu.computeMaps(predictions, output_dir / f"test_{idx}.gold")
            # dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            bleu4 = compute_bleu(references, predictions)
            meteor, rougel = compute_meteor_rouge(references, predictions)
            with open(Path(args.output_dir) / 'final_res.json', 'w') as f:
                json.dump(dict(BLEU_4=bleu4, METEOR=meteor, ROUGE_L=rougel), f, indent=4)
            logger.info(f"BLEU_4: {bleu4}\tMETEOR: {meteor}\t ROUGE_L: {rougel}")
            logger.info("  "+"*"*20)


if __name__ == "__main__":
    main()
