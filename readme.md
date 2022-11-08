# Python Script for Extracting Deep Learning Features from DTD Dataset

This script extracts well-known Deep Learning features from DTD Dataset using the models and weights in keras. Please
see the keras documentation for details on model architectures, data and training procedures used.

Currently, the script has support for Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet,
MobileNetV2, DenseNet121, DenseNet201, NASNetMobile, NASNetLarge and EfficientNetV2L and can be easily extended to
extract further networks in keras.

# Sample usage

## Downloading pre-requisite packages

Please use the requirements.txt file to install the required dependencies. It can be done with:

```
$ pip install -r requirements.txt
```

It is extremely recommended that you use a virtual environment to execute the line above.

## Usage Example

To extract features using the VGG16 architecture:

```
$ python3 main.py -a VGG16
```

The resulting structure of the `VGG16` folder will be:

```
VGG16/
   -> Fold_1.txt                  
   -> Fold_2.txt                  
   -> Fold_3.txt                  
   -> Fold_4.txt                  
   -> Fold_5.txt                  
   -> Fold_6.txt                  
   -> Fold_7.txt                  
   -> Fold_8.txt                  
   -> Fold_9.txt                  
   -> Fold_10.txt                  
   -> weights.best.hdf5                  
```

where `fold-X.txt` contains the accuracy for fold `X`. The `weights.best.hdf5  ` file, on the other hand, contains the
best weight obtained for the dataset with this architecture

## Parameters

Required arguments:

```
 -a --architecture 	        Which architecture will be used to extract the features (Xception|VGG16|VGG19|ResNet50|InceptionV3|InceptionResNetV2|MobileNet|MobileNetV2|DenseNet121|DenseNet201|NASNetMobile|NASNetLarge|EfficientNetV2L) 
```

Optional arguments:

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -p --path           |    dtd/images/           |Path where the images are found
| -s --seed          | 1994           |
| -hi -–height_image           |    300                |
| -wi --width_image               | 300               |
| -qib --quantity_images_batch                   | 30             |
| -–patience            | 3           |
| -e --epochs             | 25             |
| -lr --learning_rate          | 1e-04           |