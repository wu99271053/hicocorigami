#!/bin/bash

python hicocorigami/preprocessing.py --raw_dir=raw --window=64 --i_type=Outward --data_dir=notransform/64/64_processed
python hicocorigami/preprocessing.py --raw_dir=raw --window=32 --i_type=Outward --data_dir=notransform/32/32_processed
python hicocorigami/preprocessing.py --raw_dir=raw --window=16 --i_type=Outward --data_dir=notransform/16/16_processed

python hicocorigami/train.py --window=64 --val_chr=1 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_1
python hicocorigami/train.py --window=64 --val_chr=2 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_2
python hicocorigami/train.py --window=64 --val_chr=3 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_3
python hicocorigami/train.py --window=64 --val_chr=4 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_4
python hicocorigami/train.py --window=64 --val_chr=5 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_5
python hicocorigami/train.py --window=64 --val_chr=6 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_6
python hicocorigami/train.py --window=64 --val_chr=7 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_7
python hicocorigami/train.py --window=64 --val_chr=8 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_8
python hicocorigami/train.py --window=64 --val_chr=9 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_9
python hicocorigami/train.py --window=64 --val_chr=10 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_10
python hicocorigami/train.py --window=64 --val_chr=11 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_11
python hicocorigami/train.py --window=64 --val_chr=12 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_12
python hicocorigami/train.py --window=64 --val_chr=13 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_13
python hicocorigami/train.py --window=64 --val_chr=14 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_14
python hicocorigami/train.py --window=64 --val_chr=15 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_15
python hicocorigami/train.py --window=64 --val_chr=16 --data-root=notransform/64/64_processed/ --save_path=notransform/64/checkpoint_16


python hicocorigami/train.py --window=32 --val_chr=1 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_1
python hicocorigami/train.py --window=32 --val_chr=2 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_2
python hicocorigami/train.py --window=32 --val_chr=3 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_3
python hicocorigami/train.py --window=32 --val_chr=4 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_4
python hicocorigami/train.py --window=32 --val_chr=5 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_5
python hicocorigami/train.py --window=32 --val_chr=6 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_6
python hicocorigami/train.py --window=32 --val_chr=7 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_7
python hicocorigami/train.py --window=32 --val_chr=8 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_8
python hicocorigami/train.py --window=32 --val_chr=9 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_9
python hicocorigami/train.py --window=32 --val_chr=10 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_10
python hicocorigami/train.py --window=32 --val_chr=11 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_11
python hicocorigami/train.py --window=32 --val_chr=12 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_12
python hicocorigami/train.py --window=32 --val_chr=13 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_13
python hicocorigami/train.py --window=32 --val_chr=14 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_14
python hicocorigami/train.py --window=32 --val_chr=15 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_15
python hicocorigami/train.py --window=32 --val_chr=16 --data-root=notransform/32/32_processed/ --save_path=notransform/32/checkpoint_16



python hicocorigami/train.py --window=16 --val_chr=1 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_1
python hicocorigami/train.py --window=16 --val_chr=2 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_2
python hicocorigami/train.py --window=16 --val_chr=3 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_3
python hicocorigami/train.py --window=16 --val_chr=4 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_4
python hicocorigami/train.py --window=16 --val_chr=5 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_5
python hicocorigami/train.py --window=16 --val_chr=6 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_6
python hicocorigami/train.py --window=16 --val_chr=7 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_7
python hicocorigami/train.py --window=16 --val_chr=8 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_8
python hicocorigami/train.py --window=16 --val_chr=9 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_9
python hicocorigami/train.py --window=16 --val_chr=10 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_10
python hicocorigami/train.py --window=16 --val_chr=11 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_11
python hicocorigami/train.py --window=16 --val_chr=12 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_12
python hicocorigami/train.py --window=16 --val_chr=13 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_13
python hicocorigami/train.py --window=16 --val_chr=14 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_14
python hicocorigami/train.py --window=16 --val_chr=15 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_15
python hicocorigami/train.py --window=16 --val_chr=16 --data-root=notransform/16/16_processed/ --save_path=notransform/16/checkpoint_16

