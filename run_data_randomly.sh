export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

cd /data/share/kingxu/code/internetware

version=data_randomly
current=`date "+%Y%m%d_%H%M%S"`

lr=5e-4
epochs=50
batch_size=32
beam_size=1

hidden_size=128
num_attention_heads=4
intermediate_size=512

max_src=150

output_dir=$version/$current

# TODO: modify the path
train_file={path to dataset}/train.pkl
train_diff_file={path to dataset}/train.jsonl
dev_file={path to dataset}/valid.pkl
dev_diff_file={path to dataset}/valid.jsonl
test_file={path to dataset}/test.pkl
test_diff_file={path to dataset}/test.jsonl

src_vocab_file=./vocabulary/src_vocab.json
node_vocab_file=./vocabulary/node_vocab.json
tgt_vocab_file=./vocabulary/tgt_vocab.json

python run.py --do_train --do_eval \
    --output_dir $output_dir \
    --train_filename $train_file \
    --train_diff_file $train_diff_file \
    --dev_filename $dev_file \
    --dev_diff_file $dev_diff_file \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --train_batch_size $batch_size \
    --eval_batch_size $batch_size \
    --beam_size $beam_size \
    --src_vocab_file $src_vocab_file \
    --node_vocab_file $node_vocab_file \
    --tgt_vocab_file $tgt_vocab_file \
    --hidden_size $hidden_size \
    --num_attention_heads $num_attention_heads \
    --intermediate_size $intermediate_size \
    --max_src $max_src

saved_model=$version/$current/checkpoint-best-ppl/pytorch_model.bin # this is trained CoMeG model

python run.py --do_test \
    --load_model_path $saved_model \
    --output_dir $output_dir \
    --test_filename $test_file \
    --test_diff_file $test_diff_file \
    --eval_batch_size $batch_size \
    --beam_size $beam_size \
    --src_vocab_file $src_vocab_file \
    --node_vocab_file $node_vocab_file \
    --tgt_vocab_file $tgt_vocab_file \
    --hidden_size $hidden_size \
    --num_attention_heads $num_attention_heads \
    --intermediate_size $intermediate_size \
    --max_src $max_src
