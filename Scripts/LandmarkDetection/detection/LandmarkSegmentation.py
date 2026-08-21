""" IMPORTS """

# Utility packages for time management
import time
import datetime
# Packages for visualization
import matplotlib.pyplot as plt
import numpy as np
# Import utilities for files
import os
import json

# Packages for deep learning
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric, MSEMetric
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import SaveImage
import torch

# Custom functions and classes
from .utils.utils import TrainOverview, Chronometer, create_model_run_directory
from .data.data import retrieve_data, datasets_trainval, dataloaders_trainval, dataset_test, dataloader_test
from .model.model import landmark_detection_model
from .losses.losses import LandmarkDetectionLoss, TaskPrioritization
from .transforms.transforms import data_augmentation_transforms_lm_seg, pre_processing_transforms_lm_seg, post_processing_transforms
from .transforms.compute import extract_landmarks_from_segmentation
from .metrics.metrics import EuclideanDistanceMetric
    

""" SEED """

SEED = 23


""" GENERAL DETECTION NETWORK """

class Network():
    
    def __init__(self, dataset_directory: str, model_directory: str, create_new_model: bool = True, identification: int = -1) -> None:
        
        set_determinism(seed=SEED)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # model running directory
        self.model_run_directory, identification = create_model_run_directory(model_directory, identification, create_new_model)
        self.dataset_directory = dataset_directory

        # training overview
        self.train_overview = TrainOverview(dataset_directory, model_directory, identification)
        
        # setup device
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.train_overview.add_general("device", device_name)
        
        # chronometer for functions
        self.chronometer = Chronometer()
        
        # checks for soundness of pipeline
        self.setup_done = False
        self.train_done = False
        self.eval_done = False
        
    # Setup dataset
    def _setup_data(self, training: dict, num_processes: int = 14) -> None:
        
        # time
        self.chronometer.go(1)

        # get and set dataset directories
        dataset_directory = self.train_overview.get_general("dataset_directory")
        partition = training["partition"]
        crossvalidation = training["crossvalidation"]
        
        # check if datalist exists with correct partition of train-val and test data
        self.train_data, self.val_data, self.test_data, self.fingerprints = retrieve_data(dataset_directory, partition, crossvalidation, num_processes)
     
        # save general information
        self.train_overview.add_training("partition", partition)
        self.train_overview.add_training("crossvalidation", crossvalidation)

        # time
        data_setup_time_seconds = self.chronometer.stop(1)
        
    # Setup given model
    def _setup_model(self, model: dict) -> None:
        
        # time
        self.chronometer.go(1)
        
        # model
        self.net_det = landmark_detection_model(model)
        self.net_det.to(self.device)
        self.train_overview.set_model(model)
        self.train_overview.add_model("modality", self.fingerprints["modality"])
            
        # time
        model_setup_time_seconds = self.chronometer.stop(1)

    # Setup transforms and dataloaders
    def _setup_datasets(self, data_augmentation: dict, post_processing: dict, num_processes: int = 14) -> None:
        
        # time
        self.chronometer.go(1)
        
        # data augmentation list
        radius = self.train_overview.get_model("radius")
        self.channels_seg = self.train_overview.get_model("out_channels_seg")
        self.channels_lm = self.train_overview.get_model("out_channels_lm")
        self.merged = self.train_overview.get_model("merged")

        patch_size = None
        num_samples = 1
        if not (self.train_overview.get_model("inference") == 'Simple'):
            patch_size = self.train_overview.get_model("patch_size")
            num_samples = 2

        train_transforms, transforms_list, val_transforms = data_augmentation_transforms_lm_seg(data_augmentation, radius=radius, channels_seg=self.channels_seg, channels_lm=self.channels_lm, merged=self.merged, patch_size=patch_size, num_samples=num_samples)
        self.train_overview.add_training("data_augmentation", transforms_list)
        
        self.n_lm = len(self.channels_lm)
        if self.merged:
            self.channels_seg = [0]
        if self.channels_seg == [-1]:
            self.channels_seg = []
        self.n_seg = len(self.channels_seg)

        # make datasets
        self.train_ds, self.val_ds = datasets_trainval(self.train_data, train_transforms, self.val_data, val_transforms, num_processes, all_in_cache=False)

        # post processing
        self.post_pred, post_processing_list = post_processing_transforms(post_processing)
        self.train_overview.add_model("post_processing", post_processing_list)
        
        # time
        datasets_setup_time_seconds = self.chronometer.stop(1)
        
    # Setup training parameters
    def _setup_train(self, training: dict, num_processes: int = 14) -> None:
        
        # time
        self.chronometer.go(1)
        
        # data loaders
        batch_size = training["batch_size"]
        self.train_overview.add_training("batch_size", batch_size)
        self.train_loader, self.val_loader = dataloaders_trainval(self.train_ds, self.val_ds, batch_size, num_processes)
        
        # epochs
        amount_epochs = training["amount_epochs"]
        self.train_overview.add_training("amount_epochs", amount_epochs)
        # validation interval = at least 100 validation steps
        validation_interval = training["validation_interval"]
        min_validation_interval = int(np.ceil(amount_epochs/100))
        if min_validation_interval < validation_interval:
            print(f"WARNING: Validation interval changed from {validation_interval} to {min_validation_interval} to contain at least 100 validation steps.")
            validation_interval = min_validation_interval
        self.train_overview.add_training("validation_interval", validation_interval)
        
        # loss function
        loss_function = training["loss_function"]
        if isinstance(loss_function, str):
            loss_function = [loss_function]
        self.train_overview.add_training("loss_function", loss_function)

        geocon_config = {}
        if training["GeoCon"]:
            with open(os.path.join(self.dataset_directory, training["GeoCon"] + '.json'), 'r') as json_file:
                geocon_config = json.load(json_file)
            self.train_overview.add_training("geometric_confinement", geocon_config)
                
        self.loss_function_det = LandmarkDetectionLoss(loss_function[0], geocon_config, self.channels_lm)
 
        if len(loss_function) > 1:
            if loss_function[1] == "DiceCELoss":
                self.loss_function_seg = DiceCELoss(sigmoid=True)
            elif loss_function[1] == "DiceLoss":
                self.loss_function_seg = DiceLoss(sigmoid=True)
            elif loss_function[1] == "DiceFocalLoss":
                self.loss_function_seg = DiceFocalLoss(sigmoid=True)
            else:
                raise Exception(f"Non implemented segmentation loss function '{loss_function[1]}'.")
            self.task_prioritization = TaskPrioritization()
        else:
            self.loss_function_seg = None

        # learning rate
        learning_rate = training["learning_rate"]
        self.train_overview.add_training("learning_rate", learning_rate)
    
        # optimizer
        optimizer = training["optimizer"]
        self.train_overview.add_training("optimizer", optimizer)
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.net_det.parameters(), learning_rate)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.net_det.parameters(), learning_rate, momentum=0.99, nesterov=True)
        else:
            raise Exception(f"Non implemented optimizer '{optimizer}'.")
            
        # learning rate scheduler
        learning_rate_scheduler = training.get("learning_rate_scheduler", "none")
        if learning_rate_scheduler.lower() in ["no", "none", ""]:
            self.use_lr_scheduler = False
        elif learning_rate_scheduler == "CosineAnnealing":
            self.train_overview.add_training("learning_rate_scheduler", learning_rate_scheduler)
            self.use_lr_scheduler = True
            self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=amount_epochs, eta_min=learning_rate/1000)
        elif learning_rate_scheduler == "Polynomial":   
            self.train_overview.add_training("learning_rate_scheduler", learning_rate_scheduler)
            self.use_lr_scheduler = True
            self.learning_rate_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, power=0.9, total_iters=amount_epochs)
        else:
            raise Exception(f"Non implemented learning rate scheduler '{learning_rate_scheduler}'.")
        
        # amp gradient scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        # metric
        validation_metric = training["validation_metric"]
        if isinstance(validation_metric, str):
            validation_metric = [validation_metric]
        self.train_overview.add_training("validation_metric", validation_metric)
        
        if validation_metric[0] == "DiceMetric":
            self.metric_det = DiceMetric()
        elif validation_metric[0] == "MSEMetric":
            self.metric_det = MSEMetric()
        else:
            raise Exception(f"Non implemented validation metric '{validation_metric[0]}'.")
        
        self.metric_seg = None
        if len(validation_metric) > 1:
            if validation_metric[1] == "DiceMetric":
                self.metric_seg = DiceMetric()
            else:
                raise Exception(f"Non implemented validation metric '{validation_metric[1]}'.")

        self.check_metric = EuclideanDistanceMetric()
        
        # inferer
        inference = self.train_overview.get_model("inference")
        if inference == "Simple":
            self.inferer = SimpleInferer()
        elif inference == "SlidingWindow":
            patch_size = self.train_overview.get_model("patch_size")
            self.inferer = SlidingWindowInferer(roi_size=patch_size, overlap=0.5)
        else:
            raise Exception(f"Non implemented inference method '{inference}'.")
            
        # checkpoint
        self.checkpoint = os.path.exists(os.path.join(self.model_run_directory, "checkpoint.pt"))
        if self.checkpoint:
            self.train_overview.load(self.model_run_directory)
            checkpoint = torch.load(self.checkpoint)
            self.net_det.load_state_dict(checkpoint["state_dict"])
            self.net_det.epoch = checkpoint["epoch"]
            self.net_det.optimizer = checkpoint["optimizer"]
            
        # time
        train_setup_time_seconds = self.chronometer.stop(1)

    # Setup test dataset and dataloader
    def _setup_testing(self) -> None:
        
        # get information
        num_processes = self.train_overview.get_general("num_processes")
        radius = self.train_overview.get_model("radius")
        channels_seg = self.train_overview.get_model("out_channels_seg")
        channels_lm = self.train_overview.get_model("out_channels_lm")
        merged = self.train_overview.get_model("merged")
        batch_size = 1
        self.n_lm = len(channels_lm)
        
        # pre-processing
        test_transforms = pre_processing_transforms_lm_seg(self.fingerprints, normalize_intensity=False, radius=radius, 
                                                           channels_seg=channels_seg, channels_lm=channels_lm, merged=merged)
        
        # make dataset and data loader
        test_ds = dataset_test(self.test_data, test_transforms, 0)
        self.test_loader = dataloader_test(test_ds, batch_size, 0)
    
    # Setup training pipeline
    def setup(self, config, available_num_processes=16):
        
        if isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = json.load(f)
            config = config_dict
        
        # check if pipeline triggered correctly
        if not self.setup_done and not self.train_done and not self.eval_done:
            print("\n - SETUP - \n")
        else:
            raise Exception("Setup already done or triggered unexpectedly.")
            
        # limit number of processes to not use all of them
        num_processes = int(0.9*available_num_processes)
        self.train_overview.add_general("num_processes", num_processes)
        
        # time
        self.chronometer.go(0)
        
        # setup training pipeline
        self._setup_data(config.get('training'), num_processes)
        self._setup_model(config.get('model'))
        self._setup_datasets(config.get('data_augmentations'), config.get('post_processing'), num_processes)
        self._setup_train(config.get('training'), num_processes)
        
        # time
        setup_time_seconds = self.chronometer.stop(0)
        self.train_overview.add_general("setup_time_seconds", setup_time_seconds)
        
        # set setup done 
        self.setup_done = True
        
            
    # Estimate training time
    def estimate_train_time(self, report=True) -> None:
        
        # time
        self.chronometer.go()
        
        # get relevant information
        if report:
            print("\nEstimating training time...")
            print(" Training step...")
        amount_epochs = self.train_overview.get_training("amount_epochs")
        validation_interval = self.train_overview.get_training("validation_interval")
        image_resolution = self.train_overview.get_model("image_resolution")
        timetest_training_batches = min(len(self.train_loader), 5)
        net_det = self.net_det
        
        # initialize epoch
        epoch_loss = 0
        step = 0
        timetest_mean_batch_time = 0
        
        # training step
        net_det.train()
        for batch_data in self.train_loader:
            
            timetest_batch_time = time.time()
            
            # inputs and labels
            inputs, labels = (batch_data["image"].to(self.device),
                              batch_data["label"].to(self.device),)
            
            """
            from monai.visualize import blend_images
            
            label, _ = torch.max(batch_data["label"][0,:self.n_lm], dim=0, keepdim=True)
            ret = blend_images(image=batch_data["image"][0], label=label, alpha=0.5, rescale_arrays=False)
            for i in range(1, 12):
                # plot the slice 50 - 100 of image, label and blend result
                slice_index = 10 * i
                plt.figure("blend image and label", (12, 4))
                plt.subplot(1, 3, 1)
                plt.title(f"image slice {slice_index}")
                plt.imshow(batch_data["image"][0, 0, :, :, slice_index], cmap="gray")
                plt.subplot(1, 3, 2)
                plt.title(f"label slice {slice_index}")
                plt.imshow(label[0, :, :, slice_index])
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.title(f"blend slice {slice_index}")
                # switch the channel dim to the last dim
                plt.imshow(torch.moveaxis(ret[:, :, :, slice_index], 0, -1))
                plt.savefig(os.path.dirname(__file__) + "/" + str(step) + "_" + str(slice_index))
            """
            # prepare the gradients for this step's back propagation
            self.optimizer.zero_grad()
            
            # pass low res image through low res network
            outputs = net_det(inputs)
            
            if self.loss_function_seg:
                segs = labels[:,self.n_lm:]
                loss_det = self.loss_function_det(outputs[:,:self.n_lm], labels[:,:self.n_lm], segs)
                loss_seg = self.loss_function_seg(outputs[:,self.n_lm:], segs)
                loss = self.task_prioritization(loss_seg, loss_det)
            else:
                loss = self.loss_function_det(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # report state
            epoch_loss += loss.item()
            step += 1
            
            # time for epoch
            timetest_batch_time = time.time() - timetest_batch_time
            # only consider batch time after first one to not get influence of data loading
            if step > 1:
                timetest_mean_batch_time += timetest_batch_time
            if report:
                print(f"   Batch {step}: {timetest_batch_time:.2f} s")
            if step > timetest_training_batches:
                break
        
        # estimate epoch time
        timetest_epoch_time = self.chronometer.stop()
        # mean batch time for batches without loading of the data (batch number > 1)
        timetest_mean_batch_time /= (timetest_training_batches-1)
        # estimated time for epoch = exact time until know + estimated time to run remaining batches
        timetest_epoch_time += (len(self.train_loader) - timetest_training_batches) * timetest_mean_batch_time
        
        # evalutation step
        self.chronometer.go()
        if report:
            print(" Evaluation step...")
            
        # define best metrics
        best_metric = -1
        best_metric_epoch = -1
        metric_values = []
        
        # switch off training features of the network for this pass
        net_det.eval()
        # 'with torch.no_grad()' switches off gradient calculation for the scope of its context
        with torch.no_grad():
            
            # iterate over each batch of images and run them through the network in evaluation mode
            for val_batch_data in self.val_loader:

                # inputs and labels
                val_inputs, val_labels, val_positions = (val_batch_data["image"].to(self.device),
                                                         val_batch_data["label"].to(self.device),
                                                         val_batch_data["label_lm_positions"],)

                # prediction on validation set
                val_outputs = self.inferer(val_inputs, net_det)
                val_probs = torch.sigmoid(val_outputs)
                if self.metric_seg is not None:
                    val_probs = val_probs[:,:self.n_lm]
                centers = extract_landmarks_from_segmentation(val_probs)
                val_outputs = torch.stack([self.post_pred(i) for i in decollate_batch(val_outputs)])

                # calculate metrics
                self.check_metric(centers, val_positions)
                if self.metric_seg is not None:
                    self.metric_det(val_outputs[:,:self.n_lm], val_labels[:,:self.n_lm])
                    self.metric_seg(val_outputs[:,self.n_lm:], val_labels[:,self.n_lm:])
                else:
                    self.metric_det(val_outputs, val_labels)

            # check validation metric
            check_metric = self.check_metric.aggregate().item()
            self.check_metric.reset()
            check_metric *= image_resolution[0]
            
            # check validation metric
            val_metric = self.metric_det.aggregate().item()
            self.metric_det.reset()
            if self.metric_seg is not None:
                val_metric_seg = self.metric_seg.aggregate().item()
                self.metric_seg.reset()
                val_metric = [val_metric, val_metric_seg]
                metric_values.append(val_metric)
                val_metric = sum(val_metric)
        
            # update network if imporovement of validation metric
            if val_metric > best_metric:
                best_metric = val_metric
                best_metric_epoch = 1
        
        print()
        self.train_overview.show()
        # validation time
        timetest_validation_time = self.chronometer.stop()      
        # estimate time for training
        estimated_epoch_time = timetest_epoch_time*amount_epochs
        estimated_validation_time = timetest_validation_time*(amount_epochs//validation_interval)
        estimated_train_time_seconds = round(estimated_epoch_time + estimated_validation_time)
        estimated_train_time_hours = datetime.timedelta(seconds=estimated_train_time_seconds)
        print(f"\nEstimated time for training : {estimated_train_time_hours} hours ({int(timetest_epoch_time)} s/epoch, {int(timetest_validation_time)} s/validation)")
        self.train_overview.add_general("estimated_train_time_seconds", estimated_train_time_seconds)
        
        # check max GPU memory
        max_gpu_memory_GB = round(torch.cuda.max_memory_allocated(self.device)*1e-9,2)
        print(f"Maximal allocated memory: {max_gpu_memory_GB} GB ({int(100*max_gpu_memory_GB/40)}%)")
    
    # Save checkpoints
    def _save_checkpoint(self, epoch, model, optimizer):
        checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        checkpoint_dir = os.path.join(self.model_run_directory, "checkpoint.pt")
        torch.save(checkpoint, checkpoint_dir)
        
    # Plot training process
    def _plot_train(self):
        
        # make figure
        plt.figure(figsize=(10, 4))
        
        # loss per epoch
        plt.subplot(1, 2, 1)
        loss_function_name = self.train_overview.get_training("loss_function")
        if not isinstance(loss_function_name, list):
            loss_function_name = [loss_function_name]
        plt.title("Training loss")
        epoch_loss_values = self.train_overview.get_general("epoch_loss_values")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        plt.xlabel("epoch")
        plt.plot(x, epoch_loss_values)
        plt.legend(loss_function_name)
        
        # validation metric per epoch
        plt.subplot(1, 2, 2)
        validation_metric_name = self.train_overview.get_training("validation_metric")
        if not isinstance(validation_metric_name, list):
            validation_metric_name = [validation_metric_name]
        plt.title("Validation metric")
        metric_values = self.train_overview.get_general("metric_values")
        validation_interval = self.train_overview.get_training("validation_interval")
        x = [validation_interval * (i) for i in range(len(metric_values))]
        plt.xlabel("epoch")
        plt.plot(x, metric_values)
        plt.legend(validation_metric_name)
        
        # save and show
        plt.savefig(os.path.join(self.model_run_directory, "plot_train"))
        plt.show()
     
    #
    def _train_val_metric(self) -> None:
        self.net_det.load_state_dict(torch.load(os.path.join(self.model_run_directory, "best_model.pth")))
        validation_metric_name = self.train_overview.get_training("validation_metric")
        image_resolution = self.train_overview.get_model("image_resolution")
        
        self.net_det.eval()
        with torch.no_grad():
            channels_seg = self.train_overview.get_model("out_channels_seg")
            channels_lm = self.train_overview.get_model("out_channels_lm")
            radius = self.train_overview.get_model("radius")
            merged = self.train_overview.get_model("merged")
            train_transforms = pre_processing_transforms_lm_seg(self.fingerprints, normalize_intensity=False, radius=radius, 
                                                           channels_seg=channels_seg, channels_lm=channels_lm, merged=merged)
            train_ds = dataset_test(self.train_data, train_transforms, 0)
            train_dl = dataloader_test(train_ds, 1, 0)

            # train metric
            for train_batch_data in train_dl:

                train_inputs, train_labels, train_positions = (train_batch_data["image"].to(self.device),
                                                               train_batch_data["label"].to(self.device),
                                                               train_batch_data["label_lm_positions"])
                
                train_outputs = self.inferer(train_inputs, self.net_det)
                train_probs = torch.sigmoid(train_outputs)
                if train_probs.shape[1] > self.n_lm:
                    train_probs = train_probs[:,:self.n_lm]
                train_centers = extract_landmarks_from_segmentation(train_probs)
                train_outputs = torch.stack([self.post_pred(i) for i in decollate_batch(train_outputs)])
                
                if train_outputs.shape[1] > self.n_lm:
                    det_train_outputs = train_outputs[:,:self.n_lm]
                else:
                    det_train_outputs = train_outputs
                if train_labels.shape[1] > self.n_lm:
                    det_train_labels = train_labels[:,:self.n_lm]
                else:                    
                    det_train_labels = train_labels
                self.metric_det(det_train_outputs, det_train_labels)

                if self.metric_seg is not None:
                    self.metric_seg(train_outputs[:,self.n_lm:], train_labels[:,self.n_lm:])

                self.check_metric(train_centers, train_positions)
            
            # check validation metric
            train_dist = self.check_metric.aggregate().item()
            self.check_metric.reset()
            train_dist *= image_resolution[0]
            train_metric = self.metric_det.aggregate().item()
            self.metric_det.reset()
            if self.metric_seg is not None:
                train_metric_seg = self.metric_seg.aggregate().item()
                self.metric_seg.reset()
                train_metric = [train_metric, train_metric_seg]

            # validation metric
            for val_batch_data in self.val_loader:
                
                val_inputs, val_labels, val_positions = (val_batch_data["image"].to(self.device),
                                                         val_batch_data["label"].to(self.device),
                                                         val_batch_data["label_lm_positions"],)

                # prediction on validation set
                val_outputs = self.inferer(val_inputs, self.net_det)
                val_probs = torch.sigmoid(val_outputs)
                if val_probs.shape[1] > self.n_lm:
                    val_probs = val_probs[:,:self.n_lm]
                val_centers = extract_landmarks_from_segmentation(val_probs)
                val_outputs = torch.stack([self.post_pred(i) for i in decollate_batch(val_outputs)])

                # calculate metrics
                if val_outputs.shape[1] > self.n_lm:
                    det_val_outputs = val_outputs[:,:self.n_lm]
                else:
                    det_val_outputs = val_outputs
                if val_labels.shape[1] > self.n_lm:
                    det_val_labels = val_labels[:,:self.n_lm]
                else:                    
                    det_val_labels = val_labels
                self.metric_det(det_val_outputs, det_val_labels)

                if self.metric_seg is not None:
                    self.metric_seg(val_outputs[:,self.n_lm:], val_labels[:,self.n_lm:])

                self.check_metric(val_centers, val_positions)
            
            # check validation metric
            val_dist = self.check_metric.aggregate().item()
            self.check_metric.reset()
            val_dist *= image_resolution[0]
            val_metric = self.metric_det.aggregate().item()
            self.metric_det.reset()
            if self.metric_seg is not None:
                val_metric_seg = self.metric_seg.aggregate().item()
                self.metric_seg.reset()
                val_metric = [val_metric, val_metric_seg]

        
        # report
        print(f"Training Euclidean distance = {train_dist}")
        print(f"Validation Euclidean distance = {val_dist}\n")
        if self.metric_seg is not None:
            print(f"Training {validation_metric_name[0]} = {train_metric[0]}, {validation_metric_name[1]} = {train_metric[1]}")
            print(f"Validation {validation_metric_name[0]} = {val_metric[0]}, {validation_metric_name[1]} = {val_metric[1]}")
        else:
            print(f"Training {validation_metric_name[0]} = {train_metric}")
            print(f"Validation {validation_metric_name[0]} = {val_metric}")
            
        self.train_overview.add_general("euclidean_distance_train", train_dist)
        self.train_overview.add_general("euclidean_distance_val", val_dist)
        self.train_overview.add_general("best_metric_train", train_metric)
        self.train_overview.add_general("best_metric_val", val_metric)
        
    # Train model
    def train(self, wandb_key_directory=None, batch_report=True):
        
        # check if pipeline triggered correctly
        if self.setup_done and not self.train_done and not self.eval_done:
            print("\n - TRAINING - \n")
        else:
            raise Exception("Setup not done, training already done or triggered unexpectedly.")
            
        # time
        self.chronometer.go()

        # get relevant information
        image_resolution = self.train_overview.get_model("image_resolution")
        amount_epochs = self.train_overview.get_training("amount_epochs")
        validation_interval = self.train_overview.get_training("validation_interval")
        loss_name = self.train_overview.get_training("loss_function")
        validation_metric_name = self.train_overview.get_training("validation_metric")

        # weights and biases
        if wandb_key_directory:
            import wandb
            learning_rate = self.train_overview.get_training("learning_rate")
            dataset_name = self.train_overview.get_general("dataset_name")
            model_identification = self.train_overview.get_general("identification")
            model_type = self.train_overview.get_model("model_type")
            project = "LandmarkSegmentation"
            with open(wandb_key_directory, 'r') as file:
                key = file.read()
            wandb.login(key=key)
            wandb.init(
                project=project, 
                name=f"{dataset_name}_{model_type}{model_identification}", 
                config={
                "architecture": model_type,
                "dataset": dataset_name,
                "epochs": amount_epochs,
                "learning_rate": learning_rate,
                "loss": loss_name,
                "validation": validation_metric_name})
            wandb.alert(
                title='Run started',
                text=f'Run started',
                level=wandb.AlertLevel.WARN,
            )
            print()
        
        # show training overview
        self.train_overview.show()

        # initialize training
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()
        check_values = list()
        
        # training
        for epoch in range(amount_epochs):
            
            # initialize epoch
            print("_" * 25)
            print(f"Epoch {epoch + 1}/{amount_epochs} -> {len(self.train_loader)} batches")
            epoch_seg_loss = 0
            epoch_det_loss = 0
            step = 0
        
            # training step
            self.net_det.train()
            for batch_data in self.train_loader:
                
                # inputs and labels
                inputs, labels = (batch_data["image"].to(self.device),
                                  batch_data["label"].to(self.device),)
                
                # prepare the gradients for this step's back propagation
                self.optimizer.zero_grad()

                outputs = self.net_det(inputs)
            

                if labels.shape[1] > self.n_lm:
                    segs = labels[:, self.n_lm:]
                    if self.merged:
                        segs = torch.max(segs, dim=1, keepdim=True).values
                    det_labels = labels[:,:self.n_lm]
                else:
                    segs = None
                    det_labels = labels
                if outputs.shape[1] > self.n_lm:
                    det_outputs = outputs[:,:self.n_lm]
                    # seg_outputs = outputs[:,self.n_lm:].copy()
                else:
                    det_outputs = outputs
                    # seg_outputs = None
                loss_det = self.loss_function_det(det_outputs, det_labels, segs)

                if self.loss_function_seg is not None:
                    if segs is None:
                        raise Exception("Segmentation loss function defined but no segmentation labels provided.")
                    if segs.shape[1] > self.n_seg:
                        segs = segs[:,self.channels_seg]
                    loss_seg = self.loss_function_seg(outputs[:,self.n_lm:], segs)
                    # loss = loss_det + loss_seg
                    loss = self.task_prioritization(loss_seg, loss_det)
                else:
                    loss_seg = None
                    loss = loss_det
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self._save_checkpoint(epoch, self.net_det, self.optimizer)
                
                # update epoch loss
                epoch_det_loss += loss_det.item()
                if loss_seg is not None:
                    epoch_seg_loss += loss_seg.item()
                step += 1
                if batch_report:
                    if loss_seg is not None:
                        print(f"  B{step}:\t{loss_name[0]} = {loss_det.item():.4f}, {loss_name[1]} = {loss_seg.item():.4f})")
                    else:
                        print(f"  B{step}:\t{loss_name[0]} = {loss_det.item():.4f}")
                
                # check euclidean distance
                probs = torch.sigmoid(outputs)
                if probs.shape[1] > self.n_lm:
                    probs = probs[:,:self.n_lm]
                if labels.shape[1] > self.n_lm:
                    labels = labels[:,:self.n_lm]
                centers = extract_landmarks_from_segmentation(probs.detach())
                positions = extract_landmarks_from_segmentation(labels.detach(), binary=True)
                self.check_metric(centers, positions)
        
            # update learning rate
            if self.use_lr_scheduler:
                self.learning_rate_scheduler.step()
                    
            # calculate mean epoch loss over current step
            epoch_det_loss /= step
            epoch_loss = epoch_det_loss
            if self.loss_function_seg:
                epoch_seg_loss /= step
                epoch_loss = [epoch_det_loss, epoch_seg_loss]
            epoch_loss_values.append(epoch_loss)
            
            # check euclidean distance
            train_check_metric = self.check_metric.aggregate().item()
            self.check_metric.reset()
            train_check_metric *= image_resolution[0]
            
            # report mean loss
            if self.loss_function_seg:
                print(f"  Average {loss_name[0]} = {epoch_det_loss:.4f}, {loss_name[1]} = {epoch_seg_loss:.4f})")
            else:
                print(f"  Average {loss_name[0]} = {epoch_det_loss:.4f}")
            print(f"  Average Euclidean distance (train): {train_check_metric} mm")
        
            # after several epochs, run our metrics to evaluate it
            if (epoch % validation_interval == 0) or (epoch == amount_epochs):
                
                self.net_det.eval()
                with torch.no_grad():
                    
                    # iterate over each batch of images and run them through the network in evaluation mode
                    for val_batch_data in self.val_loader:
                        
                        # inputs and labels
                        val_inputs, val_labels, val_positions = (val_batch_data["image"].to(self.device),
                                                                 val_batch_data["label"].to(self.device),
                                                                 val_batch_data["label_lm_positions"],)

                        # prediction on validation set
                        val_outputs = self.inferer(val_inputs, self.net_det)

                        # get center coordinates of landmark outputs
                        val_probs = torch.sigmoid(val_outputs)
                        if val_probs.shape[1] > self.n_lm:
                            val_probs = val_probs[:,:self.n_lm]
                        val_centers = extract_landmarks_from_segmentation(val_probs).cpu()

                        # postprocessing outputs to segmentation maps
                        val_outputs = torch.stack([self.post_pred(i) for i in decollate_batch(val_outputs)])
                        
                        # calculate metrics
                        self.check_metric(val_centers, val_positions)
                        if val_outputs.shape[1] > self.n_lm:
                            det_val_ouputs = val_outputs[:,:self.n_lm]
                        else:
                            det_val_ouputs = val_outputs
                        if val_labels.shape[1] > self.n_lm:
                            det_val_labels = val_labels[:,:self.n_lm]
                        else:
                            det_val_labels = val_labels
                        self.metric_det(det_val_ouputs, det_val_labels)

                        if self.metric_seg is not None:
                            self.metric_seg(val_outputs[:,self.n_lm:], val_labels[:,self.n_lm:])

                    # check validation metric
                    val_check_metric = self.check_metric.aggregate().item()
                    self.check_metric.reset()
                    val_check_metric *= image_resolution[0]
                    check_values.append([train_check_metric, val_check_metric])
                    val_metric_det = self.metric_det.aggregate().item()
                    self.metric_det.reset()
                    val_metric = val_metric_det
                    if self.metric_seg is not None:
                        val_metric_seg = self.metric_seg.aggregate().item()
                        self.metric_seg.reset()
                        val_metric = [val_metric, val_metric_seg]
                    metric_values.append(val_metric)
                    
                    # update network if imporovement of validation metric
                    if self.metric_seg is not None:
                        val_metric = self.n_lm*val_metric[0] + val_metric[1]
                    if val_metric > best_metric:
                        best_metric = val_metric
                        best_metric_epoch = epoch + 1
                        torch.save(self.net_det.state_dict(), os.path.join(self.model_run_directory, "best_model.pth"))
                        
                    # train overview checkpoint
                    max_gpu_memory_GB = round(torch.cuda.max_memory_allocated(self.device)*1e-9,2)
                    self.train_overview.add_general("max_gpu_memory_GB", max_gpu_memory_GB)
                    self.train_overview.add_general("epoch_loss_values", epoch_loss_values)
                    self.train_overview.add_general("metric_values", metric_values)
                    self.train_overview.add_general("check_values", check_values)
                    self.train_overview.save()
                        
                    # weights and biases report
                    if wandb_key_directory:
                        metrics = {"train/train_det_loss": epoch_det_loss,
                                   "train/epoch": epoch + 1,
                                   "train/learning_rate": self.learning_rate_scheduler.get_last_lr()[0],
                                   "train/mEuclidDist(mm)": train_check_metric,
                                   "val/mEuclidDist(mm)": val_check_metric,
                                   "val/val_det_metric": val_metric_det,}
                        if self.loss_function_seg:
                            metrics.update({"train/train_seg_loss": epoch_seg_loss})
                        if self.metric_seg is not None:
                            metrics.update({"val/val_seg_metric": val_metric_seg})
                        wandb.log(metrics)
                        
                    # report
                    print(f"Current average {' + '.join(validation_metric_name)} = {val_metric:.4f}")
                    print(f"Current average Euclidean distance (val)= {round(val_check_metric, 3)} mm")
                    print(f"Best average {' + '.join(validation_metric_name)} = {best_metric:.4f}",
                          f"at epoch {best_metric_epoch}")
                    
        torch.save(self.net_det.state_dict(), os.path.join(self.model_run_directory, "last_model.pth"))
        
        # real training time
        print(f"\nTraining completed. Best {' + '.join(validation_metric_name)} = {best_metric:.4f}",
              f"at epoch {best_metric_epoch}")
        real_train_time_seconds = self.chronometer.stop()
        self.train_overview.add_general("real_train_time_seconds", real_train_time_seconds)
        real_train_time_hours = datetime.timedelta(seconds=real_train_time_seconds)
        print(f"Time for training : {real_train_time_hours} hours\n")
        
        # end training
        if wandb_key_directory:
            wandb.finish()
        self._plot_train()
        self._train_val_metric()
        self.train_overview.save()
        
        # set training done
        self.train_done = True

    # Load trained model
    def load_state(self, model_id_directory: str, best: bool = True) -> None:
        
        # load train overview of model
        self.train_overview.load(model_id_directory)
        
        # get dataset
        num_processes = self.train_overview.get_general("num_processes")
        self._setup_data(self.train_overview.get_training(), num_processes=num_processes)
        
        # get model
        self._setup_model(self.train_overview.get_model())
        
        # load best model
        self.model_run_directory = model_id_directory
        if best:
            self.net_det.load_state_dict(torch.load(os.path.join(self.model_run_directory, "best_model.pth")))
        else:
            self.net_det.load_state_dict(torch.load(os.path.join(self.model_run_directory, "last_model.pth")))
        
        # load inferer
        inference = self.train_overview.get_model("inference")
        if inference == "Simple":
            self.inferer = SimpleInferer()
        elif inference == "SlidingWindow":
            patch_size = self.train_overview.get_model("patch_size")
            self.inferer = SlidingWindowInferer(roi_size=patch_size, overlap=0.5)
        
        # load post processing
        post_processing_list = self.train_overview.get_model("post_processing")
        self.post_pred, _ = post_processing_transforms(post_processing_list)
        
        # load validation metric
        validation_metric = self.train_overview.get_training("validation_metric")
        
        if validation_metric[0] == "DiceMetric":
            self.metric_det = DiceMetric()
        elif validation_metric[0] == "MSEMetric":
            self.metric_det = MSEMetric()
        else:
            raise Exception(f"Non implemented validation metric '{validation_metric[0]}'")
        
        self.metric_seg = None
        if len(validation_metric)>1:
            if validation_metric[1] == "DiceMetric":
                self.metric_seg = DiceMetric()
            else:
                raise Exception(f"Non implemented validation metric '{validation_metric[1]}'")
                
        self.check_metric = EuclideanDistanceMetric()
            
        # overwrite train overview to go back to original one (original setup times overwritten during setups)
        self.train_overview.load(model_id_directory)
        
        # set setup and training done
        self.setup_done = True
        self.train_done = True
        if "test_metric" in self.train_overview.get_general().keys():
            self.eval_done = False
        
    # Evaluate model   
    def evaluate(self) -> None:
        
        # check if pipeline triggered correctly
        if self.setup_done and self.train_done:
            print("\n - EVALUATION - \n")
        else:
            raise Exception("Setup not done, training not done or evaluation triggered unexpectedly.")
        
        # load test data
        self._setup_testing()

        # get relevant information
        figure_directory = os.path.join(self.model_run_directory, "plot_evaluate")
        if not os.path.isdir(figure_directory):
            os.makedirs(figure_directory)
        validation_metric_name = self.train_overview.get_training("validation_metric")
        image_resolution = self.train_overview.get_model("image_resolution")
        
        outputs_directory = os.path.join(self.model_run_directory, "outputs")
        if not os.path.isdir(outputs_directory):
            os.makedirs(outputs_directory)
            
        self.net_det.eval()

        pred = []
        gt = []
        
        inference_time = 0
        inference_memory = 0
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated(self.device)

        mean_intensity = self.fingerprints["mean_intensity"]
        std_intensity = self.fingerprints["std_intensity"]
        p_0_5_intensity = self.fingerprints["p_0_5"]
        p_99_5_intensity = self.fingerprints["p_99_5"]
        from monai.transforms import Compose, ThresholdIntensity, NormalizeIntensity
        intensity = Compose([ThresholdIntensity(threshold=p_0_5_intensity, 
                                 above=True, 
                                 cval=p_0_5_intensity),
             ThresholdIntensity(threshold=p_99_5_intensity, 
                                 above=False, 
                                 cval=p_99_5_intensity),
             NormalizeIntensity(subtrahend=mean_intensity, 
                                 divisor=std_intensity)])
        
        # evaluation
        with torch.no_grad():
            
            for i, test_batch_data in enumerate(self.test_loader):
                
                start_time = time.time()
                
                test_inputs_real, test_labels, test_positions = (test_batch_data["image"].to(self.device),
                                                            test_batch_data["label"].to(self.device),
                                                            test_batch_data["label_lm_positions"],)

                test_inputs = intensity(test_inputs_real)
                # if i == 4 or i == 5:
                #     test_inputs = torch.flip(test_inputs, dims=[-3])
                # prediction on validation set
                test_outputs = self.inferer(test_inputs, self.net_det)
                print(test_outputs.shape, test_labels.shape)

                # get center coordinates of landmark outputs
                test_probs = torch.sigmoid(test_outputs)
                if test_probs.shape[1] > self.n_lm:
                    test_probs = test_probs[:,:self.n_lm]
                test_centers = extract_landmarks_from_segmentation(test_probs)
                
                test_outputs = torch.stack([self.post_pred(i) for i in test_outputs])
                
                inference_time += (time.time() - start_time)
                inference_memory = (torch.cuda.max_memory_allocated(self.device) - initial_memory)/(1000**3)

                if test_outputs.shape[1] > self.n_lm:
                    det_test_outputs = test_outputs[:,:self.n_lm]
                else:
                    det_test_outputs = test_outputs
                if test_labels.shape[1] > self.n_lm:
                    det_test_labels = test_labels[:,:self.n_lm]
                else:                    
                    det_test_labels = test_labels
                self.metric_det(det_test_outputs, det_test_labels)

                if self.metric_seg is not None:
                    self.metric_seg(test_outputs[:,self.n_lm:], test_labels[:,self.n_lm:])

                self.check_metric(test_centers, test_positions)
                pred.extend(test_centers.detach().cpu().tolist())
                gt.extend(test_positions.detach().cpu().tolist())

                # save outputs
                for batch_number in range(len(test_outputs)):
                    
                    # plot output vs label
                    identifier = i*len(test_outputs)+batch_number+1
                    channels_seg = self.train_overview.get_model("out_channels_seg")
                    if channels_seg == [-1]:
                        channels_seg = []
                    num_channels = len(self.train_overview.get_model("out_channels_seg"))

                    if self.metric_seg is not None:
                        plt.figure(figsize=(12, 18))
                        pos = int(test_positions.cpu()[batch_number,0,2].item())
                        test_images_slice = test_inputs_real.cpu()[batch_number, -1, :, :, pos]
                        test_labels_slice = torch.sum(test_labels.cpu()[batch_number, self.n_lm:], dim=0)[:, :, pos]
                        test_outputs_slice = torch.sum(test_outputs[batch_number].cpu()[self.n_lm:], dim=0)[:, :, pos]
                        plt.subplot(1, 2, 1)
                        plt.imshow(test_images_slice, cmap="gray")
                        plt.imshow(np.where(test_labels_slice==0, np.nan, test_labels_slice))
                        plt.title(f"Label {identifier}")
                        ax = plt.subplot(1, 2, 2)
                        plt.imshow(test_images_slice, cmap="gray")
                        plt.imshow(np.where(test_outputs_slice==0, np.nan, test_outputs_slice), alpha=0.5, cmap="Blues_r")
                        plt.title(f"Output {identifier}")
                        plt.show()
                        saver = SaveImage(output_dir=os.path.join(outputs_directory, str(identifier)), output_postfix="seg", separate_folder=False)
                        saver(test_outputs[batch_number][self.n_lm:])
                        plt.savefig(f"{figure_directory}/{identifier}_seg.png")
                        plt.show()

                    num_channels = len(self.train_overview.get_model("out_channels_lm"))

                    plt.figure(figsize=(12, 18))
                    for i in range(num_channels):
                        pos = test_positions.cpu()[batch_number,i]
                        pred_pos = test_centers.cpu()[batch_number,i]
                        # move the data to the CPU
                        slice_number = int(pos[2].item())
                        test_images_slice = test_inputs_real.cpu()[batch_number, 0, :, :, slice_number]
                        test_labels_slice = test_labels.cpu()[batch_number, i, :, :, slice_number]
                        test_probs_slice = test_probs[batch_number].cpu()[i][:, :, slice_number]
                        test_outputs_slice = test_outputs[batch_number].cpu()[i][:, :, slice_number]
                        ax = plt.subplot(num_channels, 2, 2*i+1)
                        ax.imshow(test_images_slice, cmap="gray")
                        ax.imshow(np.where(test_labels_slice==0, np.nan, test_labels_slice), alpha=0.5, cmap='Greens_r')
                        ax.scatter([int(pos[1].item())], [int(pos[0].item())], marker='o', c='green')
                        plt.title(f"Label {identifier}")
                        ax = plt.subplot(num_channels, 2, 2*i+2)
                        ax.imshow(test_images_slice, cmap="gray")
                        ax.imshow(np.where(test_probs_slice<=0.001, np.nan, test_probs_slice), alpha=0.5, cmap="Reds")
                        ax.scatter([int(pos[1].item())], [int(pos[0].item())], marker='o', c='green')
                        ax.scatter([int(pred_pos[1].item())], [int(pred_pos[0].item())], marker='.', c='red')
                        plt.title(f"Output {identifier} (channel {i})")
                        plt.show()
                        saver = SaveImage(output_dir=os.path.join(outputs_directory, str(identifier)), output_postfix=f"lm{i+1}", separate_folder=False)
                        saver(test_probs[batch_number][i])
                    plt.savefig(f"{figure_directory}/{identifier}_pos.png")
                    plt.show()
                    

        # get validation metric on test dataset
        inference_time /= len(self.test_loader)
        check_metric = self.check_metric.aggregate().item()
        self.check_metric.reset()
        check_metric *= image_resolution[0]
        test_metric = self.metric_det.aggregate().item()
        self.metric_det.reset()
        if self.metric_seg is not None:
            test_metric_seg = self.metric_seg.aggregate().item()
            self.metric_seg.reset()
            test_metric = [test_metric, test_metric_seg]
                
        print(f"\nInference time = {inference_time:.2f} s, GPU memory = {inference_memory} GB\n")
        if self.metric_seg is not None:
            print(f"Testing {validation_metric_name[0]} = {test_metric[0]}, {validation_metric_name[1]} = {test_metric[1]}")
        else:
            print(f"Testing {validation_metric_name[0]} = {test_metric}")
        self.train_overview.add_general("raw_results", [pred, gt])
        self.train_overview.add_general("euclidean_distance_test", check_metric)
        self.train_overview.add_general("test_metric", test_metric)
        self.train_overview.add_general("inference_memory", inference_memory)
        self.train_overview.add_general("inference_time", inference_time)
        self.train_overview.save()
        
        # set evaluation done
        self.eval_done = True
        
        
        
if __name__ == '__main__':
    
    dataset_directory = ""
    model_directory = ""
    config_directory = ""
    
    
    network = Network(dataset_directory, model_directory)
    
    network.setup(config_directory)
    
    network.train()
    
    network.evaluate()
