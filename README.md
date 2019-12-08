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

module load python/3.7.1 (any python > 3.6.0 will do)

(python3.6 -m pip3 install -U --user pip setuptools virtualenv) #necessary for first time virtualenv setup

python3.6 -m venv .env --system-site-packages    # --system-site-packages seems to be key to not get version collisions

source .env/bin/activate

pip install nlp-architect==0.5.1

(pip install (--user) torchvision==0.3.0) #if necessary

(pip install (--user) torch==1.3.1) #if necessary

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
 
 bsub -o ~/tmp/rte-8bit/ -W 24:00 -n 1 -R "rusage[mem=8196]" < 0.5.1trainRTE.sh

