# Pipeline design and before springbreak plan
## Christian Roncal, CMSC499A

# Pipeline design:
## Files:
4 main python files: Data, Engine, Train, Models

### 1. Train
Entry point. Where settings dict will be defined and where the Data / dataloader, model, and Engine objects will be instantiated. 

**Requirements:**
1. Be able to define settings dict
1. Be able to use different models/data

### 2. Engine
Contains the training loops for training, evaluation and testing. 

**Requirements:**
1. Different types of models/dataloaders should be able to use this
1. Training loops will just be called in Train

### 3. Data
Pretty much a file with torch Dataset classes for Single task and Mulitask to be used to create data loaders. Has a class which processes raw data and converts it into dataset into a dataloader.

### 4. Models
Not really a file more like multiple files, each model will be defined as its own file. We're looking to test different CF techniques in the future if time permits.

## Flow 
In the train.py file:
1. raw csv is passed to DatasetGenerator object which processes data to create Torch dataset into dataloader.
1. Loop for epochs:
    1. Training data dataloader will be passed into a training loop
    1. Evaluate on validation data
    1. Plot to tensorboard.
    
    
# Plan before springbreak
1. Actually learn AUC-PR. Got started reading, and have a basic idea. I found a knowledgebase about evaluations
https://classeval.wordpress.com. Maybe write some notes ~ 2 days

1. Clarify about the data and possibly repair data loading (add negatives)
~ 2 days

1. Finish evaluation code for data and engine
~ 2 days

1. Run single task tests ~ 1 day

1. Create datagenerator, and GMF model for multitask ~ rest of the days