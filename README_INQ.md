# How to run:
1. clone https://github.com/Mxbonn/INQ-pytorch
2. Write one of the two quantization schedulers to $INQ_ROOT/inq/quantization_scheduler.py
3. create venv
4. pip install -e $INQ_ROOT/
5. clone transformers
6. Copy run_glue_inq.py into $HUGGINGFACE_ROOT/transformets/examples
7. pip install -r requirements.txt (requirements.txt in transformers/examples I think)
8. Run run_glue_inq.py like this:
    python ./run_glue_inq.py \\
    --model_type bert   \\   
    --model_name_or_path bert-base-uncased      \\
    --task_name $TASK_NAME      \\
    --do_train      \\
    --do_eval      \\
    --do_lower_case     \\
    --data_dir $GLUE_DIR/$TASK_NAME      \\
    --max_seq_length 128     \\
    --per_gpu_eval_batch_size=8      \\  
    --per_gpu_train_batch_size=8     \\  
    --learning_rate 2e-5     \\
    --num_train_epochs 9.0    \\  
    --save_steps 100000      \\
    --output_dir $OUT_DIR  \\
    --overwrite_output_dir
    
   # On leonhard
   
   module load python_gpu/3.7.4
   
   module clone INQ-pytorch transformers QuantizedBERT_DeepLearningProject
   
   change QuantizedBERT_DeepLearningProject Branch to INQ
   
   copy quantization_scheduler.py into INQ-Pytorch/inq
   
   python -m venv .inq_env --system-site-packages
   
   enter venv
   
   pip install -e INQ-pytorch
   
   cp QuantizedBERT_DeepLearningProject/run_glue_inq.py to /transformers/examples/
   
   pip install -r transformers/examples/requirements.txt
   
   pip install -e transformers
   
   Run run_glue_inq.py like this:
   
   python ./transformers/examples/run_glue_inq.py \\
   
    --model_type bert   \\   
    --model_name_or_path bert-base-uncased      \\
    --task_name $TASK_NAME      \\
    --do_train      \\
    --do_eval      \\
    --do_lower_case     \\
    --data_dir $GLUE_DIR/$TASK_NAME      \\
    --max_seq_length 128     \\
    --per_gpu_eval_batch_size=8      \\  
    --per_gpu_train_batch_size=8     \\  
    --learning_rate 2e-5     \\
    --num_train_epochs 9.0    \\  
    --save_steps 100000      \\
    --output_dir $OUT_DIR  \\
    --overwrite_output_dir
   
    
