# How to run:
1. Write one of the two quantization schedulers to $INQ_ROOT/inq/quantization_scheduler.py
2. run pip intall -e $INQ_ROOT/
3. Copy run_glue_inq.py into $HUGGINGFACE_ROOT/transformets/examples/
4. Run run_glue_inq.py like this:
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
    
