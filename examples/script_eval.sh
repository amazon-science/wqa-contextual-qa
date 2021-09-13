python run_eval.py \
       --model ../../models/contextual/asnq/electra-base-discriminator.local-ord.asnq.lr:1e+05-p:2-b:512-e:15-s:256-w:2-r:1.pt \
       --context local-ord \
       --cuda 0 \
       --test ../data/squad2.0-dev.csv \
       --workers 8 \
       --batch_size 128 \
       --max_seq_length 256 \
       --debug
