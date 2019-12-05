# QuantizedBERT_DeepLearningProject

Install from source (Github)
git clone https://github.com/NervanaSystems/nlp-architect.git
cd nlp-architect
pip install -e .  # install in developer mode

implement quantized bert, if error:

  pip install python-dateutil==2.6
  pip uninstall transformers 
  pip install transformers==2.0
  
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

