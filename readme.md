# Emotion Analysis Text-Classification Project

Welcome to the Emotion Analysis project, a text-classification endeavor that focuses on analyzing emotions from textual input.

## Branch Setup

Before you begin, make sure to switch to the `onnx` branch which is set up for running training with ONNX models:

```commandline
git checkout onnx
```


## Training

To run the training part of the project, install the required dependencies and then start the training process with the following commands:
```commandline
pip install -r requirements.txt
python main.py
```


If you wish to train the model using LoRA, run:
```commandline
python peftTrainner.py
```


### AWS Training Setup

If you plan to train the model on AWS, you'll need to mount your S3 bucket by running:
```commandline
mount_s3.sh
```


## Results

After training, you can find the results in `result.csv` for the standard training and `lora_result.csv` for the LoRA-trained model. All detailed data and parameters are saved in the `eval` folder.

## Deployment

To deploy the model and start the server, run:
```commandline
python run.py
```

You can then navigate to `localhost:5001` in your web browser to access the interface where you can input text for emotion analysis testing.
