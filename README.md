# InvisionAI_PixelClassifier
Python Exercise for InvisionAI interview process
   
1. Install all the dependencies from requirements.text in your python3 environment
    
    pip3 install -r requirements.txt
    
2. Create the datasets for training and testing by running

    python3 generate_dataset.py
    
    The above script creates dataset in the following structure
    
    rooftop/\
    ├── test/\
    │   ├── data/\
    │   │   ├── 00004.png\
    │   │   ├── 00005.png\
    │   │   ├── ...\
    │   └── label/\
    │       ├── 00004.png\
    │       ├── 00005.png\
    │       ├── ...\
    └── train/\
        ├── data/\
        │   ├── 00000.png\
        │   ├── 00001.png\
        │   ├── ...\
        └── label/\
            ├── 00000.png\
            ├── 00001.png\
            ├── ...
            
3. To train the PixelClassification CNN model to detect rooftops in aerial view images, run
   
    python3 train.py
    
    The trained model will be saved in the following structure
    
    models/\
    ├── model_v1.pth\
    ├── model_v2.pth\
    ├── model_v3.pth\
    ├── model_v4.pth\
    └── model_v5.pth
    
4. To test, evaluate the trained model, visualize and save the output, run

    python3 prediction.py
    
    The output is saved by defaul to ./output.png
    
5. Files like data.py, utils.py , models.py support the execution of the project