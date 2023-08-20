mkdir experiments/baseline-plus-augms-manymodels/model-1; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-1 MODEL_KWARGS.dim 64
mkdir experiments/baseline-plus-augms-manymodels/model-2; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-2 MODEL_KWARGS.dim 128
mkdir experiments/baseline-plus-augms-manymodels/model-3; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-3 MODEL_KWARGS.dim 256
mkdir experiments/baseline-plus-augms-manymodels/model-4; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-4 MODEL_KWARGS.dim 384
mkdir experiments/baseline-plus-augms-manymodels/model-5; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-5 MODEL_KWARGS.dim 512
mkdir experiments/baseline-plus-augms-manymodels/model-6; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-6 MODEL_KWARGS.dim 64 MODEL_KWARGS.head_size 32
mkdir experiments/baseline-plus-augms-manymodels/model-7; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-7 MODEL_KWARGS.dim 128 MODEL_KWARGS.head_size 32
mkdir experiments/baseline-plus-augms-manymodels/model-8; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-8 MODEL_KWARGS.dim 256 MODEL_KWARGS.head_size 32
mkdir experiments/baseline-plus-augms-manymodels/model-9; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-9 MODEL_KWARGS.dim 384 MODEL_KWARGS.head_size 32
mkdir experiments/baseline-plus-augms-manymodels/model-10; python train.py --config-file config.json OUT experiments/baseline-plus-augms-manymodels/model-10 MODEL_KWARGS.dim 512 MODEL_KWARGS.head_size 32