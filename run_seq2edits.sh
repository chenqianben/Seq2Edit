
# 模型训练: bert/distilbert/albert/roberta/xlnet 
# 示例：
# bash eval_seq2edits.sh bert 0

# SRC_FILE=./outputs/test.src
# TGT_FILE=./outputs/seq2edits_$model_name.out
# M2_FILE=./outputs/seq2edits_$model_name.m2

SRC_FILE=$1
TGT_FILE=$2
M2_FILE=$3

model_name=xlnet
root_dir=/home/LAB/luopx/bea2019/gector
data_dir=$root_dir/data_seq2edits

if [ $model_name = "roberta" ]; then # roberta模型中使用的特殊符号不是[CLS] [SEP]
    special_tokens_fix=1
else
    special_tokens_fix=0
fi


echo "eval $model_name on GPU $device, special_tokens_fix: $special_tokens_fix"

# bert模型 inference
python3 $root_dir/predict.py --model_path $root_dir/ckpts/seq2edits_$model_name/model_state_epoch_14.th \
                  --vocab_path $data_dir/output_vocabulary \
                  --input_file $SRC_FILE \
                  --output_file $TGT_FILE --transformer_model $model_name  \
                  --special_tokens_fix $special_tokens_fix


# score
echo Writing Score result to $M2_FILE
python2 $HOME/bea2019/fairseq/m2scorer/m2scorer -v \
    $TGT_FILE /home/LAB/luopx/bea2019/data/conll14st-test-data/noalt/official-2014.combined.m2 > $M2_FILE

