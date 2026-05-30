if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FinancialForecasting" ]; then
    mkdir ./logs/FinancialForecasting
fi

seq_len=96
pred_len=8
model_name=PathFormer

root_path_name=./dataset/finance
data_path_name=fslr.csv

model_id_name=fslr

data_name=custom

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data_name \
  --features M \
  --target close \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --num_nodes 6 \
  --num_workers 0 \
  --layer_nums 3 \
  --patch_size_list 16 12 8 4  \
                          12 8 6 4 \
                          8 6 2 12 \
  --residual_connection 1 \
  --k 2 \
  --d_model 16 \
  --d_ff 64 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.0005 \
  --batch_size 32 \
  --itr 1 \
  > logs/FinancialForecasting/fslr.log