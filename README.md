# Drought-forecasting-from-satellite-image-using-ML

This uses deep learning (CNN Efficient Net model) to predict drought conditions from satellite images and produces a simple-to-use webapp through which users can get predictions for their uploaded satellite images.

# Data

The current dataset consists of 86,317 train and 10,778 validation satellite images, 65x65 pixels each, in 10 spectrum bands, with 10,774 images withheld to test long-term generalization (107,869 total). Human experts (pastoralists) have labeled these with the number of cows that the geographic location at the center of the image could support (0, 1, 2, or 3+ cows). Each pixel represents a 30 meter square, so the images at full size are 1.95 kilometers across. Pastoralists are asked to rate the quality of the area within 20 meters of where they are standing, which corresponds to an area slightly larger a single pixel. Since forage quality is correlated across space, the larger image may be useful for prediction.

# Model

The model was developed and trained using TensorFlow 2. We use CNNs based on an EfficientNet algorithm (https://github.com/google/automl/tree/master/efficientnetv2#readme). EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Efficient net models use neural architecture search (NAS) to jointly optimize model size and training speed, and are scaled up in a way for faster training and inference speed.

# Drought prediction
