import utils

config = utils.EasyDict({
	'task': 'classification',
	# 'task': 'regression',

	'uncertainty': False,

	# 'dataset_dir': '../DementiaBank'
	'dataset_dir': '../ADReSS-IS2020-data/train',
	'test_dataset_dir': '../ADReSS-IS2020-data/test',

	# 'model_dir': 'models/non_uncertainty_classification/1',
	# 'model_types': ['compare'],

	# 'model_dir': 'models/uncertainty_boosting_rmse/1',
	# 'model_dir': 'models/testing_silence/',
	# 'model_dir': 'models/contrastiveLearning_boosting_rmse',
	'model_dir': 'models/contrastiveLearning_classification',


	'model_types': ['intervention'],
	# 'model_types': [ 'silences'],

	# 'training_type': 'bagging',
	'training_type' :'boosting',

	'boosting_type': 'rmse',
	# 'boosting_type': 'stddev',


	'n_folds': 5,

	'dataset_split' :'full_dataset',
	# 'dataset_split' :'k_fold',
	'split_ratio': 0.8,

	'voting_type': 'hard_voting',
	# 'voting_type': 'soft_voting',
	# 'voting_type': 'learnt_voting',
	# 'voting_type': 'uncertainty_voting',

	'plot': True,
    
	'longest_speaker_length': 32,
	'n_pause_features': 11,
	'compare_features_size': 21,
	'split_reference': 'samples',

	'n_epochs': 2000,
	'batch_size': 24,
	'lr': 0.001,
	'verbose': 0,

	#Data Augmentation
	'isAugment_acoustic': True,
	'isAugment_intervention': True,
	'isAugment_disfluency': True,
	
	'isAugment_acoustic_Test': True,
	'isAugment_intervention_Test': True,
	'isAugment_disfluency_Test': True,

	'isContrastive': True,
	'contrastiveW': 0.5


})
