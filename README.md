# BDRL
# The encoder of the BDRL
![image](https://github.com/Anonymous-author-code/BDRL/blob/main/BDRL1.png)
# The encoder layer of the transformer in BDRL
![image](https://github.com/Anonymous-author-code/BDRL/blob/main/Attention1.png)
# Illustrations of fine-tuning BDRL for solution classification
![image](https://github.com/Anonymous-author-code/BDRL/blob/main/Fine-turning%201.png)
# Data
Most of the data sets in the experiment are generated data, such as VRP, TSP, etc.
# How to run   
BDRL can be divided into three independent models: BDRL as a whole, BDRL without fine-tuning, and BERT fine-tuning only.
# Generating data
python generate_data.py --Data all --name validation --seed 
python generate_data.py --Data all --name test --seed 
# Tranining
python run.py --graph_size 20 --run_name 'tsp20' --val_dataset data/tsp/tsp20_validation_seed4321.pkl
# Test
python eval.py data/tsp/tsp20_test_seed.pkl --model pretrained/tsp_20 --decode_strategy sample --width 1280 --eval_batch_size 1
# Fine-tuning
# VRP
python run_bert_classifier.py --task_name co --do_train --do_eval --do_predict --data_dir ./data/vrp --bert_model bert-base-uncased --max_seq_length 20 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_vrp/ --gradient_accumulation_steps 1 --eval_batch_size 512
# TSP
python run_bert_classifier.py --task_name co --do_train --do_eval --do_predict --data_dir ./data/tsp --bert_model bert-base-cased --max_seq_length 200 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir ./output_tsp/ --gradient_accumulation_steps 1 --eval_batch_size 512
