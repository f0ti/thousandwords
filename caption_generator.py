"""
It will first convert all the images from the sample_images folder into a set of VGG16 features, 
and then pass the features to the trained deep-learning model and tokenizer, and return the captions.
"""

import os
import keras
import numpy as np
from build_model.build_model import feature_extractions, sample_caption
from pickle import load, dump
    
# Load tokenizer
def infer():
	with open("./tokenizer.pkl", "rb") as f:
		tokenizer = load(f)
    
	model = keras.models.load_model("model/model0_v19_devloss_3.38_trainloss_3.1.h5") #Load model
	vocab_size = tokenizer.num_words # The number of vocabulary
	max_length = 33 # Maximum length of caption sequence

	#sampling
	features = feature_extractions("./sample_images")

	captions = []
	for i, filename in enumerate(features.keys()):
			caption = sample_caption(model, tokenizer, max_length, vocab_size, features[filename])
			captions.append(caption)

	return captions

if __name__ == "__main__":
	infer()
