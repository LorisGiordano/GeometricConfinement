""" IMPORTS """

from detection.utils.parsers import det_evaluate_parser
from detection.utils.handle_slurms import get_slurm_evaluate, save_slurm


""" RUN """

if __name__ == '__main__':
    
    # Load parameters
    args = det_evaluate_parser()
    # slurm_evaluate = get_slurm_evaluate(args.root)
    
    # Setup network
    match args.task:
        case 'LandmarkDetection':
            print("\nDETECTION TRAIN OUTPUT\n")
            from detection.LandmarkSegmentation import Network
            
        case _:
            raise Exception(f"Invalid task '{args.task}'")
    
    network = Network("_", args.model_directory, identification=args.id)
    network.load_state(args.model_id_directory)
    
    # Evaluate
    network.evaluate()
    
    # Save evaluate output file
    # save_slurm(slurm_evaluate, network.model_run_directory)