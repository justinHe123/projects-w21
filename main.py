import argparse
import os
import time


import constants
from datasets.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet

SUMMARIES_PATH = "training_summaries"


def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}

    # Create path for training summaries
    label = f"cassava__{int(time.time())}"
    summary_path = f"{SUMMARIES_PATH}/{label}"
    os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!
    count = args.count
    # count = 1000
    # count = 64
    train_prop = 0.70
    path = f"{args.path}/train.csv"
    data = np.genfromtxt(path, delimiter=',', dtype='str')

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
        with torch.no_grad():
            x = torch.zeros([1, 3, image_size, image_size])
            x = EfficientNet.from_pretrained(args.arch).extract_features(x)
            flatten_size = x.shape[1]*x.shape[2]*x.shape[3]            
            print(x.shape)
    else:
        image_size = args.image_size
        flatten_size = args.flatten_size
    
    print(image_size)
    print(flatten_size)
    train_dataset = StartingDataset(truth = data[1:int(count*train_prop), 1], images = data[1:int(count*train_prop), 0], base = f'{args.path}/train_images', size=image_size)
    val_dataset = StartingDataset(truth = data[int(count*train_prop):count, 1], images = data[int(count*train_prop):count, 0], base = f'{args.path}/train_images', size=image_size)

    model = StartingNetwork(3, 5, arch=args.arch, flatten_size = flatten_size)
    model = model.to(device)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
        device=device,
        model_name=args.model_name
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument(
        "--n_eval", type=int, default=constants.N_EVAL,
    )
    parser.add_argument(
        "--model_name", type=str, default=constants.MODEL_NAME,
    )
    parser.add_argument(
        "--arch", type=str, default=constants.ARCH
    )
    parser.add_argument(
        "--image_size", type=int, default=224
    )
    parser.add_argument(
        "--flatten_size", type = int, default = 0
    )
    parser.add_argument(
        "--path", type=str, default='./cassava-leaf-disease-classification'
    )
    parser.add_argument(
        "--count", type=int, default=21397
    )
    parser.add_argument(
        "--prop", type=float, default=0.70
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
