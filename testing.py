# This function takes in the test data and returns the results
from get_phrase_tags import individual_phrase_tags
from extract_features import extract_features
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

def testing(data, model, pca, scaler, apply_pca):
	ind_tags_phrases = individual_phrase_tags(data)
	ind_phrases, ind_tags, phrase_position = [], [], []
	for i in ind_tags_phrases:
		ind_phrases.append(i[0].split(',')[1])
		ind_tags.append(i[0].split(',')[0])
		phrase_position.append(i[1])
	test_features = extract_features(ind_phrases, phrase_position, 'testing')
	if apply_pca == 'y':
		test_features = scaler.transform(test_features)
		test_features = pca.transform(test_features)

	pred_tags = model.predict(test_features)
	accuracy = metrics.accuracy_score(ind_tags, pred_tags)
	return accuracy
