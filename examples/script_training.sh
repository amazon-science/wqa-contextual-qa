python run_training.py \
       --model google/electra-base-discriminator \
       --context base \
       --cuda 0 \
       --training ../data/squad2.0-train.csv \
       --validation ../data/squad2.0-dev.csv \
       --max_epochs 10 \
       --workers 4 \
       --run 0 \
       --batch_size 32 \
       --learningrate 1e-5 \
       --max_seq_length 256 \
       --output_file my-finetuned-model.pt \
       --patience 2 \
       --warmup_peak 0.5 \
       --val_metric p@1
       ###--debug
