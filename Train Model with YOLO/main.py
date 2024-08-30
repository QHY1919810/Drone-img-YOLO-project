'''
Script to train and evaluate model
'''

from ultralytics import YOLO
import argparse
import os
import yaml
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'


# python fine_tune.py 
if __name__ == "__main__":

    # build a new model from YAML
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", help="'Train' or 'Evaluate' or 'Predict")
    arg_parser.add_argument("--config", default='config.yaml', help="'Train' or 'Evaluate'")
    args = arg_parser.parse_args()

    # load config and create model 
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # loading a pretrained yolo model
    if cfg['pretrained']:
        model = YOLO(cfg['model'])     # load a pretrained model (recommended for training)
    else: model = YOLO('yolov8l.yaml') # or train from scratch
    model.to('cuda')

    # train or validate depending on argument
    if args.mode == 'Train':
        model.train(**cfg)
    elif args.mode == 'Evaluate':
        model.val(**cfg)
        # model.predict(...)  # NOTE: to get predictions rather than compute evaluation results
    else:
        exit('Input Train or Evaluate')