### Description
* config contain dataset path, pretrained model and fc layer, preprocessing
* every train, eval, inference file will read config.py and run script based on what you config so you need to config to use corresponding model as the train script you use<br>
* eval.py file will save result in folder, result contain confusion matrix and roc curve
* augmenter.py is class that store all the augment function, augment_data.py is a script that will lookup into dataset and call augmenter to augment data

## dataset folder structure
|---> train <br>
|---| --> class1 <br>
|---| --> class2 <br>
|---| --> class3 <br>
| --> test <br>
|---| --> class1 <br>
|---| --> class2 <br>
|---| --> class3 <br>
|---> eval <br>
|---| --> class1 <br>
|---| --> class2 <br>
|---| --> class3 <br>
