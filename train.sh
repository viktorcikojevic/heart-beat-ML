# Training script meant to reproduce the results.

python train.py --config-file experiments/baseline/config.json 
python train.py --config-file experiments/baseline-plus-weightedsampler/config.json 
python train.py --config-file experiments/baseline-plus-augms/config.json 



for i in {1..4}
do
    echo $i
    mkdir experiments/baseline-plus-augms-manymodels/model-$i
    python train.py --config-file experiments/baseline-plus-augms-manymodels/config.json OUT experiments/baseline-plus-augms-manymodels/model-$i TEST_FOLDS [$i,9,10]
done


for i in {5..8}
do
    echo $i
    mkdir experiments/baseline-plus-augms-manymodels/model-$i
    python train.py --config-file experiments/baseline-plus-augms-manymodels/config.json OUT experiments/baseline-plus-augms-manymodels/model-$i TEST_FOLDS [$i,9,10] USE_WEIGHTED_SAMPLER true
done