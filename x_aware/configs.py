import os


def prepare_config_and_experiments(args):
    config = init_config(args)
    config = create_experiment_name(config, args)

    # create names
    config['log_dir'] = os.path.join(args.experiments_path, config['name'])
    best_model_dir = os.path.join(args.experiments_path, 'best_models')
    config['best_model_name'] = os.path.join(best_model_dir, config['name'])  # +'.h5'

    # create folder
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])
    if not os.path.exists(config['best_model_name']):
        os.makedirs(config['best_model_name'])

    # execute experiment parameters
    config['params_train'] = {'batch_size': config['BATCH_SIZE'],
                              'shuffle': True,
                              'num_workers': 6}
    config['params_val'] = {'batch_size': config['BATCH_SIZE'],
                            'shuffle': True,
                            'num_workers': 6}
    config['params_test'] = {'batch_size': 4,
                             'shuffle': False,
                             'num_workers': 6}

    return config


def init_config(args):
    config = vars(args)

    config['model_name'] = 'InceptionResNetV2'
    config['feature_types'] = []
    config['MAX_EPOCHS'] = 20
    config['PATIENCE'] = 3
    config['BATCH_SIZE'] = 32
    config['LEARNING_RATE'] = 0.0001
    config['TEST'] = False
    config['STORE_THRESHOLD'] = 69.0
    config['TRACKING_MEASURE'] = args.val_measure
    config['FACE_PATH'] = args.face_path
    config['FEATURE_PATH'] = '../preprocessed/'
    config['ROOT_DATA_PATH'] = '../data'
    if args.testing_switch:
        config['TEST'] = True
    if config['TEST']:
        config['BATCH_SIZE'] = 16  # 32 --> 12 steps
        config['DECAY_STEPS'] = 3
        config['STORE_THRESHOLD'] = 5.

    config['FREEZE'] = None  # {epoch: 0, layers_from_front : 0, layers_from_back : 10}
    config['UNFREEZE'] = None

    if args.lr_decay_batch:
        if config['TEST']:
            config['batch_lr_adjustment'] = {'steps_non_defined': 1, 'defined': {1: 1, 2: 2}}
        else:
            config['batch_lr_adjustment'] = {'steps_non_defined': 250, 'defined': {1: 250, 2: 350}}
        print('Set :', config['batch_lr_adjustment'])
    else:
        config['batch_lr_adjustment'] = None  # {'steps_non_defined': 250,'defined': {1:250, 2:350}}

    if args.freeze_unfreeze == 'core':
        print('set freeze_unfreeze', args.freeze_unfreeze)
        config['FREEZE'] = {'epochs': 0, 'layers_from_bottom': 2, 'layers_from_top': 0}
        config['UNFREEZE'] = None
    elif args.freeze_unfreeze == 'coreunfreeze':
        print('set freeze_unfreeze', args.freeze_unfreeze)
        # in combination with no transfered weights
        config['FREEZE'] = {'epochs': 0, 'layers_from_bottom': 2, 'layers_from_top': 0}
        config['UNFREEZE'] = {'epochs': 1, 'layers_from_bottom': 2, 'layers_from_top': 0}
        # full core
    elif args.freeze_unfreeze == 'coreunfreeze2':
        print('set freeze_unfreeze', args.freeze_unfreeze)
        # in combination with no transfered weights
        config['FREEZE'] = {'epochs': 0, 'layers_from_bottom': 2, 'layers_from_top': 0}
        config['UNFREEZE'] = {'epochs': 2, 'layers_from_bottom': 2, 'layers_from_top': 0}
        # full core
    elif args.freeze_unfreeze == 'freeze2':
        print('set freeze_unfreeze', args.freeze_unfreeze)
        # in combination with no transfered weights
        config['FREEZE'] = {'epochs': 2, 'layers_from_bottom': 2, 'layers_from_top': 0}
        config['UNFREEZE'] = None
        # full core
    else:
        print('no freezing or type not known', args.freeze_unfreeze)
        config['FREEZE'] = None  # {epoch: 0, layers_from_front : 0, layers_from_back : 10}
        config['UNFREEZE'] = None
    if args.data_augmentation:
        if args.data_augmentation == 'all':
            config['random_crop'] = True
            config['random_hflip'] = True
            config['random_vflip'] = True
        elif args.data_augmentation == 'crop':
            config['random_crop'] = True
            config['random_hflip'] = False
            config['random_vflip'] = False
        elif args.data_augmentation == 'flips':
            config['random_crop'] = False
            config['random_hflip'] = True
            config['random_vflip'] = True
    else:
        config['random_crop'] = False
        config['random_hflip'] = False
        config['random_vflip'] = False
    return config


def create_experiment_name(config, args):
    name = ''
    if args.inputs == 'env_i' or args.inputs == 'all_i':
        name = name + 'env_img_'
        config['feature_types'].append('env_img_extractor')
    if args.inputs == 'face_i' or args.inputs == 'all_i':
        name = name + 'faces_img_'
        config['feature_types'].append('faces_img_extractor')
    if args.inputs == 'face_f':
        name = name + 'faces_fea_'
        config['feature_types'].append('face_extractor')
    if args.inputs == 'all':
        name = name + 'all_'
        config['feature_types'] = ['env_img_extractor', 'faces_img_extractor', 'face_extractor', 'gocar']
    name = name + config['model_name']

    if args.base_trainable:
        name = name + '_btrainable'
    else:
        name = name + '_bfrozen'
    name = name + '_' + args.head
    if args.lr_decay_batch:
        name = name + '_batchdecay'
    if args.freeze_unfreeze:
        if args.freeze_unfreeze != 'unfreeze':
            name = name + '_' + args.freeze_unfreeze
    if args.pretrained != 'imagenet':
        name = name + '_' + args.pretrained
    if config['TRACKING_MEASURE'] != 'val_acc':
        name = name + '_' + config['TRACKING_MEASURE']
    if args.data_augmentation:
        name = name + '_' + args.data_augmentation
    if args.testing_switch:
        name = name + '_T'

    config['name'] = name
    return config
