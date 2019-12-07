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
module load python/3.6.1
(python3.6 -m pip3 install -U --user pip setuptools virtualenv) #not sure if necessary, doesn't hurt
python3.6 -m venv .env --system-site-packages    # --system-site-packages seems to be key to not get version collisions
source .env/bin/activate
pip install nlp-architect==0.5.1
(pip install --user torchvision==0.3.0) #if necessary
(pip install --user torch==1.3.1) #if necessary
nano v0.5.1train.sh

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
 
 chmod 777 v0.5.1train.sh
 ./v0.5.1train.sh

