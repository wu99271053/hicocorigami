#!/bin/bash

python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_1 --val_chr=1 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_2 --val_chr=2 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_3 --val_chr=3 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_4 --val_chr=4 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_5 --val_chr=5 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_6 --val_chr=6 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_7 --val_chr=7 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_8 --val_chr=8 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_9 --val_chr=9 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_10 --val_chr=10 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_11 --val_chr=11 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_12 --val_chr=12 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_13 --val_chr=13 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_14 --val_chr=14 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_15 --val_chr=15 --data-root=gaussian/32/32_processed/
python hicocorigami/train.py --window=32 --save_path=gaussian/32/checkpoint_16 --val_chr=16 --data-root=gaussian/32/32_processed/

python hicocorigami/preprocessing.py --raw_dir=raw --window=16 --i_type=Outward --data_dir=gaussian/16/16_processed
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_1 --val_chr=1 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_2 --val_chr=2 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_3 --val_chr=3 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_4 --val_chr=4 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_5 --val_chr=5 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_6 --val_chr=6 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_7 --val_chr=7 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_8 --val_chr=8 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_9 --val_chr=9 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_10 --val_chr=10 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_11 --val_chr=11 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_12 --val_chr=12 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_13 --val_chr=13 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_14 --val_chr=14 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_15 --val_chr=15 --data-root=gaussian/16/16_processed/
python hicocorigami/train.py --window=16 --save_path=gaussian/16/checkpoint_16 --val_chr=16 --data-root=gaussian/16/16_processed/



