# Multi-Task Learning for Depression Detection in Dialogs

This is the source code repository for the paper Multi-Task Learning for Depression Detection in Dialogs (SIGDial 2022).

## Requirements
- allennlp >= 2.0
- pytorch
- numpy


## Datasets
### DAIC-WOZ
Our main task depression detection uses DAIC-WOZ (part of the Distress Analysis Interview Corpus) (Gratch et al.,2014). Download is available [here](https://dcapswoz.ict.usc.edu).

### DailyDialog
Our auxiliary tasks use Dailydialog (Li et al., 2017). Download from [here](http://yanran.li/dailydialog.html).

## Source code structure

- `main.py`: choose MODE for train and test, modify arguments for different multi-task settings
- `model.py`: hierarchical structure modeling
- `dataset_reader.py`: read daic-woz and dailydialog
- `utility.py`: store auxiliary functions
- `constant.py`: store hard-coded paths, labels, etc.

## Citation
