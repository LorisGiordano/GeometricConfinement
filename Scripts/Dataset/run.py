""" IMPORTS """

from dataset.remodeling import remodel_dataset
from dataset.utils import create_metadata
from dataset.analysis import analyze_dataset
from dataset.parsers import dataset_parser

    
""" RUN """

if __name__ == '__main__':
    
    # Load parameters
    args = dataset_parser()
    config = args.config
    dataset_directory = args.dataset_directory

    # Remodeling
    resampling = config.get('resampling')
    resizing = config.get('resizing')
    if (resampling or resizing):
        target_spacing = tuple(config.get('target_spacing'))
        target_size = tuple(config.get('target_size'))
        mode = tuple(config.get('mode'))
        dataset_directory = remodel_dataset(dataset_directory, resampling, target_spacing, resizing, target_size, mode)
    
    # Metadata
    if config.get('analysis'):
        create_metadata(dataset_directory, config)
        analyze_dataset(dataset_directory)