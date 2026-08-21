""" IMPORTS """

from detection.utils.parsers import det_time_parser


""" RUN """

if __name__ == '__main__':

    # Load parameters
    args = det_time_parser()
    
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

        case 'sebast':
            from detection.LandmarkSegmentation import Network
            
        case _:
            from detection.LandmarkSegmentation import Network
            # raise Exception(f"Invalid configuration '{args.config_name}'")

    network = Network(args.dataset_directory, args.model_directory, create_new_model=False)
    network.setup(args.config, available_num_processes=16)
    
    # Estimate training time
    network.estimate_train_time()