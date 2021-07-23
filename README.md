# InvisionAI_PixelClassifier
Python Exercise for InvisionAI interview process - segmenting the roofs of buildings in aerial view images
   
1. Install all the dependencies from requirements.text in your python3 environment
    
    ```python
    pip3 install -r requirements.txt
    ```
    
2. Create the datasets for training and testing by running

    ```python
    python3 generate_dataset.py
    ```
    
    The above script creates dataset in the following structure
    
    ```bash
    rooftop/
    ├── test/
    │   ├── data/
    │   │   ├── 00004.png
    │   │   ├── 00005.png
    │   │   ├── ...
    │   └── label/
    │       ├── 00004.png
    │       ├── 00005.png
    │       ├── ...
    └── train/
        ├── data/
        │   ├── 00000.png
        │   ├── 00001.png
        │   ├── ...
        └── label/
            ├── 00000.png
            ├── 00001.png
            ├── ...
    ```
            
3. To train the PixelClassification CNN model to detect rooftops in aerial view images, run
    
    ```python
    python3 train.py
    ```
    
    The trained model will be saved in the following structure
    
    ```bash
    models/
    ├── model_v1.pth
    ├── model_v2.pth
    ├── model_v3.pth
    ├── model_v4.pth
    └── model_v5.pth
    ```
    
4. To test, evaluate the trained model, visualize and save the output, run

    ```python
    python3 prediction.py
    ```
    
    The output is saved by defaul to ./output.png
    
5. Files like data.py, utils.py , models.py support the execution of the project


Notes:\
    1. Adam optimizer and Binary Cross-entropy are used for simplicity. Other loss functions like Binary Cross-entropy with DICE loss, IOU loss etc can also be used for better performance.\
    2. The prescribed architecture was used for modelling although the architecture could be made efficient by adding/modifying layers (BatchNorm, experimenting with different hyper-parameters etc.)\
    3. Binary accuracy could also be used as a metric but is ignore and only loss is used as the primary metric.\
    4. Several other data augmentation strategy could also be incorporated internally as pytorch transforms but have to be customized and implemented so that both the input and label have the same transformation/ augmentation applied. \
    5. During final prediction, pooling is used externally to mimic dilation so that the roof segmentations aren't spotty. It can be disabled or ignored or can be replaced by image processing techniques from OpenCV/Scikit-Image/Pillow\