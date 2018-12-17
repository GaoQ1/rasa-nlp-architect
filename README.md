## 介绍
这个repository主要是实现了intel的nlp-architect的intent和slot的提取，两个model
 - mtl
 - seq2seq

然后配合提供一个rasa-nlu的server，配合rasa-core使用。

## Train mtl model
```
python train_mtl_model.py --dataset_path rasa_data/rasa_nlu_data/ -b 100 -e 10
```

## Train seq2seq model
```
python train_seq2seq_model.py --dataset_path rasa_data/rasa_nlu_data/ -b 100 -e 10
```

## Interactive
```
python interactive.py --model_path models/mtl/model.h5 --dataset_path rasa_data/rasa_nlu_data/
```
or
```
python interactive.py --model_path models/seq2seq/model.h5 --dataset_path rasa_data/rasa_nlu_data/
```

## Provide a server
```
python server.py -m models/mtl/model.h5 -i models/mtl/model_info.dat -w logs/mtl.log
```
or
```
python server.py -m models/seq2seq/model.h5 -i models/seq2seq/model_info.dat -w logs/seq2seq.log
```

## TODO
现阶段Intent只返回了概率最大的，后续需要做intent ranking.
