# QuantizedBERT_DeepLearningProject

Install from source (Github)
git clone https://github.com/NervanaSystems/nlp-architect.git
cd nlp-architect
pip install -e .  # install in developer mode

implement quantized bert, if error:

  pip install python-dateutil==2.6
  
  pip uninstall transformers 
  
  pip install transformers==2.0
  
  possibly pip install wheel & tensorflow==2.0.0b0

TRAINING

create train.sh
new:
nlp-train transformer_glue \
    --task_name mrpc \
    --model_name_or_path bert-base-uncased \
    --model_type quant_bert \
    --learning_rate 2e-5 \
    --output_dir /tmp/mrpc-8bit \
    --evaluate_during_training \
    --data_dir /path/to/MRPC \
    --do_lower_case
    
old:
nlp_architect train transformer_glue \
    --task_name cola \
    --model_name_or_path bert-base-uncased \
    --model_type quant_bert \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir tmp/cola-8bit \
    --evaluate_during_training \
    --data_dir glue_data/CoLA \
    --do_lower_case \
    --overwrite_output_dir

EVALUATION

create eval.sh
new:
nlp-inference transformer_glue \
    --model_path /tmp/mrpc-8bit \
    --task_name mrpc \
    --model_type quant_bert \
    --output_dir /tmp/mrpc-8bit \
    --data_dir /path/to/MRPC \
    --do_lower_case \
    --overwrite_output_dir

old:
nlp_architect run transformer_glue \
    --model_path tmp/cola-8bit \
    --task_name cola \
    --model_type quant_bert \
    --output_dir tmp_out/cola-8bit \
    --data_dir glue_data/CoLA \
    --do_lower_case \
    --overwrite_output_dir \
    --evaluate
    
# Training on a cluster

rsync -Pav ~/deeplearning/glue_data/ odermafr@euler.ethz.ch:/cluster/home/odermafr/glue_data

ssh -Y odermafr@euler.ethz.ch

NOTE: using module avail python you will see, that before this command python/3.7.1 is not available, however after loading python3.6.0 using module avail python again we see new higher options

module load python/3.6.0 (these commands need to be redone on every startup of the euler)
module load python/3.6.1 (you can only load python 3.6.1 once you have loaded python 3.6.0 (I dont know why)
module unload python/3.6.0

(python3.6 -m pip3 install -U --user pip setuptools virtualenv) #necessary for first time virtualenv setup

python3.6 -m venv .env --system-site-packages    # --system-site-packages seems to be key to not get version collisions

source .env/bin/activate

pip install nlp-architect==0.5.1

pip install (--user) torchvision==0.3.0 # necessary

(pip install (--user) torch==1.3.1) #probably not necessary, rerun pip install nlp-architect==0.5.1 to check if necessary

nano v0.5.1trainCoLA.sh

INSERT THIS TEXT

    nlp_architect train transformer_glue \
        --task_name cola \
        --model_name_or_path bert-base-uncased \
        --model_type quant_bert \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --output_dir ~/tmp/cola-8bit \
        --evaluate_during_training \
        --data_dir ~/glue_data/CoLA \
        --do_lower_case
 
 chmod 777 v0.5.1trainCoLA.sh
 
 ./v0.5.1trainCoLA.sh
 
 or as a job f.ex:
 
 bsub -W 24:00 -n 1 -R "rusage[mem=8196]" < 0.5.1trainRTE.sh
 
 **Why nlp-architect==0.5.2 currently not used**
 
 because of 
 `AttributeError: 'QuantizedBertLayer' object has no attribute 'is_decoder'` 

 
 # Capabilities of nlp-architect
 
 v.0.5.1
 run
 - nlp_architect
 - nlp_architect train
 - nlp_architect train transformer_glue -h
 
 for different levels of help
 
 ALL FLAGS:
 
 nlp_architect train transformer_glue  
                                            --task_name TASK_NAME    
                                            --data_dir DATA_DIR --model_type  
                                            {bert,quant_bert,xlnet,xlm}  
                                            --output_dir OUTPUT_DIR  
                                            [--tokenizer_name TOKENIZER_NAME]  
                                            [--max_seq_length MAX_SEQ_LENGTH]  
                                            [--cache_dir CACHE_DIR]  
                                            [--do_lower_case]  
                                            [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]  
                                            [--no_cuda]  
                                            [--overwrite_output_dir]  
                                            [--overwrite_cache]  
                                            --model_name_or_path  
                                            MODEL_NAME_OR_PATH  
                                            [--config_name CONFIG_NAME]  
                                            [--evaluate_during_training]  
                                            [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]  
                                            [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]  
                                            [--learning_rate LEARNING_RATE]  
                                            [--weight_decay WEIGHT_DECAY]  
                                            [--adam_epsilon ADAM_EPSILON]  
                                            [--max_grad_norm MAX_GRAD_NORM]  
                                            [--num_train_epochs NUM_TRAIN_EPOCHS]  
                                            [--max_steps MAX_STEPS]  
                                            [--warmup_steps WARMUP_STEPS]  
                                            [--logging_steps LOGGING_STEPS]  
                                            [--save_steps SAVE_STEPS]  
                                            [--eval_all_checkpoints]  
                                            [--seed SEED]

