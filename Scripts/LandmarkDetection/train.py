""" IMPORTS """

from detection.utils.parsers import det_train_parser, save_config
from detection.utils.handle_slurms import get_slurm_train, save_slurm


""" RUN """

if __name__ == '__main__':

    # Load parameters
    args = det_train_parser()
    # slurm_train = get_slurm_train(args.config, args.model_root)
    
    # Setup network
    match args.config_name.replace("config_", "")[:6]:
        case 'lm_seg':
            print("\nLANDMARK SEGMENTATION TRAIN OUTPUT\n")
            from detection.LandmarkSegmentation import Network
        
        case 'hm_reg':
            print("\nHEATMAP REGRESSION TRAIN OUTPUT\n")
            from detection.HeatmapRegression import Network
            
        case 'lm_reg':
            print("\nLANDMARK REGRESSION TRAIN OUTPUT\n")
            from detection.LandmarkRegression import Network
            
        case _:
            from detection.LandmarkSegmentation import Network
            # raise Exception(f"Invalid configuration '{args.config_name}'")

    print(f"Short description of run: {args.short_description}\n")
    network = Network(args.dataset_directory, args.model_directory, identification=args.continue_training)
    network.setup(args.config, available_num_processes=16)
    
    # Save config file
    save_config(args.config_filepath, network.model_run_directory)
    
    # Train and evaluate
    network.train(wandb_key_directory=f"{args.model_root}/_wandb_key.txt")
    network.evaluate()
    
    # Save train output file
    # save_slurm(slurm_train, network.model_run_directory)