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
from monai.transforms import SaveImage, Activations
import torch

# Custom functions and classes
from .utils.utils import TrainOverview, Chronometer, create_model_run_directory
from .data.data import retrieve_data, datasets_trainval, dataloaders_trainval, dataset_test, dataloader_test
from .model.model import landmark_detection_model
from .losses.losses import LandmarkDetectionLoss
from .transforms.transforms import data_augmentation_transforms_hm_reg, pre_processing_transforms_hm_reg, post_processing_transforms
from .transforms.compute import extract_landmarks_from_heatmaps
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
        
        # training overview
        self.train_overview = TrainOverview(dataset_directory, model_directory, identification)
        
        # setup device
        device_name = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"
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
        sigma = self.train_overview.get_model("sigma")
        gamma = self.train_overview.get_model("gamma")
        if gamma > 1.0:
            self.activation = False
        num_channels = self.train_overview.get_model("out_channels")
        train_transforms, transforms_list, val_transforms = data_augmentation_transforms_hm_reg(data_augmentation, sigma=sigma, gamma=gamma, num_channels=num_channels)
        self.train_overview.add_training("data_augmentation", transforms_list)
        
        # make datasets
        self.train_ds, self.val_ds = datasets_trainval(self.train_data, train_transforms, self.val_data, val_transforms, num_processes)
        # remove list of train and validation filepaths
        del self.train_data
        del self.val_data

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
        
        self.loss_function_det = LandmarkDetectionLoss(loss_function[0],  training["ShaPr"], training["SurfCon"])
        
        self.loss_function_seg = None
        if len(loss_function) > 1:
            if loss_function[1] == "DiceCELoss":
                self.loss_function_seg = DiceCELoss(sigmoid=True)
            elif loss_function[1] == "DiceLoss":
                self.loss_function_seg = DiceLoss(sigmoid=True)
            elif loss_function[1] == "DiceFocal":
                self.loss_function_seg = DiceFocalLoss(sigmoid=True)
            else:
                raise Exception(f"Non implemented segmentation loss function '{loss_function[1]}'.")
            
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
                self.metric_det = DiceMetric()
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
            
        # time
        train_setup_time_seconds = self.chronometer.stop(1)

    # Setup test dataset and dataloader
    def _setup_testing(self) -> None:
        
        # get information
        num_processes = self.train_overview.get_general("num_processes")
        sigma = self.train_overview.get_model("sigma")
        gamma = self.train_overview.get_model("gamma")
        num_channels = self.train_overview.get_model("out_channels")
        batch_size = 1
        
        # pre-processing
        test_transforms = pre_processing_transforms_hm_reg(self.fingerprints, normalize_intensity=True, sigma=sigma, gamma=gamma, num_channels=num_channels)
        
        # make dataset and data loader
        test_ds = dataset_test(self.test_data, test_transforms, 0)
        del self.test_data
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
        self._setup_train(config.get('training'))
        
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
            inputs, labels, positions = (batch_data["image"].to(self.device),
                              batch_data["label"].to(self.device),
                              batch_data["label_lm_positions"])

            # prepare the gradients for this step's back propagation
            self.optimizer.zero_grad()
            
            # pass low res image through low res network
            outputs = net_det(inputs)
            
            if self.activation:
                probs = torch.sigmoid(outputs)

            if self.loss_function_seg:
                loss_det = self.loss_function_det(outputs[:-1], labels[:-1])
                loss_seg = self.loss_function_seg(outputs[-1], labels[-1])
                loss = loss_det + loss_seg
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
                val_outputs = self.net_det(val_inputs)
                if self.activation:
                    val_outputs = torch.sigmoid(val_outputs)
                val_centers = extract_landmarks_from_heatmaps(val_outputs)

                # calculate metrics
                self.check_metric(val_centers, val_positions)
                if self.metric_seg:
                    self.metric_det(val_outputs[:-1], val_labels[:-1])
                    self.metric_seg(val_outputs[-1], val_labels[-1])
                else:
                    self.metric_det(val_outputs, val_labels)

            # check validation metric
            check_metric = self.check_metric.aggregate().item()
            self.check_metric.reset()
            check_metric *= image_resolution[0]
            
            # check validation metric
            val_metric = self.metric_det.aggregate().item()
            self.metric_det.reset()
            if self.metric_seg:
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
        
            # train metric
            for train_batch_data in self.train_loader:

                train_inputs, train_labels = (train_batch_data["image"].to(self.device),
                                              train_batch_data["label"].to(self.device),)
                
                # prediction on validation set
                train_outputs = self.net_det(train_inputs)
                if self.activation:
                    train_outputs = torch.sigmoid(train_outputs)
                train_centers = extract_landmarks_from_heatmaps(train_outputs)
                train_positions = extract_landmarks_from_heatmaps(train_labels)
                
                self.check_metric(train_centers, train_positions)
                if self.metric_seg:
                    self.metric_det(train_outputs[:-1], train_labels[:-1])
                    self.metric_seg(train_outputs[-1], train_labels[-1])
                else:
                    self.metric_det(train_outputs, train_labels)
            
            # check validation metric
            train_dist = self.check_metric.aggregate().item()
            self.check_metric.reset()
            train_dist *= image_resolution[0]
            train_metric = self.metric_det.aggregate().item()
            self.metric_det.reset()
            if self.metric_seg:
                train_metric_seg = self.metric_seg.aggregate().item()
                self.metric_seg.reset()
                train_metric = [train_metric, train_metric_seg]

            # validation metric
            for val_batch_data in self.val_loader:

                val_inputs, val_labels, val_positions = (val_batch_data["image"].to(self.device),
                                                         val_batch_data["label"].to(self.device),
                                                         val_batch_data["label_lm_positions"],)

                # prediction on validation set
                val_outputs = self.net_det(val_inputs)
                if self.activation:
                    val_outputs = torch.sigmoid(val_outputs)
                val_centers = extract_landmarks_from_heatmaps(val_outputs)
                
                self.check_metric(val_centers, val_positions)
                if self.metric_seg:
                    self.metric_det(val_outputs[:-1], val_labels[:-1])
                    self.metric_seg(val_outputs[-1], val_labels[-1])
                else:
                    self.metric_det(val_outputs, val_labels)
            
            # check validation metric
            val_dist = self.check_metric.aggregate().item()
            self.check_metric.reset()
            val_dist *= image_resolution[0]
            val_metric = self.metric_det.aggregate().item()
            self.metric_det.reset()
            if self.metric_seg:
                val_metric_seg = self.metric_seg.aggregate().item()
                self.metric_seg.reset()
                val_metric = [val_metric, val_metric_seg]

        
        # report
        print(f"Training Euclidean distance = {train_dist}")
        print(f"Validation Euclidean distance = {val_dist}\n")
        if self.metric_seg:
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
        amount_epochs = self.train_overview.get_training("amount_epochs")
        validation_interval = self.train_overview.get_training("validation_interval")
        loss_name = self.train_overview.get_training("loss_function")
        validation_metric_name = self.train_overview.get_training("validation_metric")
        image_resolution = self.train_overview.get_model("image_resolution")

        # weights and biases
        if wandb_key_directory:
            import wandb
            learning_rate = self.train_overview.get_training("learning_rate")
            dataset_name = self.train_overview.get_general("dataset_name")
            model_identification = self.train_overview.get_general("identification")
            model_type = self.train_overview.get_model("model_type")
            model_directory = self.train_overview.get_general("model_directory")
            project = model_directory.replace(os.path.dirname(wandb_key_directory), "").split("/")[1]
            with open(wandb_key_directory, 'r') as file:
                key = file.read()
            wandb.login(key=key)
            wandb.init(
                project=project, 
                name=f"{model_type}{model_identification}", 
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
        task_priorities = list()
        
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

                # pass low res image through low res network
                outputs = self.net_det(inputs)
                
                # calculate loss
                if self.loss_function_seg:
                    loss_det = self.loss_function_det(outputs[:,:-1], labels[:,:-1])
                    loss_seg = self.loss_function_seg(outputs[:,-1:], labels[:,-1:])
                    loss = loss_det + loss_seg
                else:
                    loss_det = self.loss_function_det(outputs, labels)
                    loss_seg = 0
                    loss = loss_det + loss_seg
                
                # back-propagation and optimizer step
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # update epoch loss
                epoch_det_loss += loss_det.item()
                if self.loss_function_seg:
                    epoch_seg_loss += loss_seg.item()
                step += 1
                if batch_report:
                    if self.loss_function_seg:
                        print(f"  B{step}:\t{loss_name[0]} = {loss_seg.item():.4f}, {loss_name[1]} = {loss_det.item():.4f})")
                    else:
                        print(f"  B{step}:\t{loss_name[0]} = {loss_det.item():.4f}")
        
            # update learning rate
            if self.use_lr_scheduler:
                self.learning_rate_scheduler.step()
                    
            # calculate mean epoch loss over current step
            epoch_det_loss /= step
            epoch_loss = epoch_det_loss
            if self.loss_function_seg:
                epoch_seg_loss /= step
                epoch_loss = [epoch_loss, epoch_seg_loss]
            epoch_loss_values.append(epoch_loss)
            
            # report mean loss
            if self.loss_function_seg:
                print(f"  Average {loss_name[0]} = {epoch_det_loss:.4f}, {loss_name[1]} = {epoch_seg_loss:.4f})")
            else:
                print(f"  Average {loss_name[0]} = {epoch_det_loss:.4f}")
        
            # after several epochs, run our metrics to evaluate it
            if (epoch % validation_interval == 0) or (epoch == amount_epochs):
                
                self.net_det.eval()
                with torch.no_grad():
                    
                    # iterate over each batch of images and run them through the network in evaluation mode
                    for b, val_batch_data in enumerate(self.val_loader):
                        
                        # inputs and labels
                        val_inputs, val_labels, val_positions = (val_batch_data["image"].to(self.device),
                                                                 val_batch_data["label"].to(self.device),
                                                                 val_batch_data["label_lm_positions"],)

                        # prediction on validation set
                        val_outputs = self.net_det(val_inputs)
                        if self.activation:
                            val_outputs = torch.sigmoid(val_outputs)
                            val_centers = extract_landmarks_from_heatmaps(val_outputs)
                        else:
                            val_centers = extract_landmarks_from_heatmaps(val_outputs)
                        
                        # calculate metrics
                        self.check_metric(val_centers, val_positions)
                        if self.metric_seg:
                            self.metric_det(val_outputs[:-1], val_labels[:-1])
                            self.metric_seg(val_outputs[-1], val_labels[-1])
                        else:
                            self.metric_det(val_outputs, val_labels)

                        inputs = val_inputs.cpu().numpy()
                        val_probs = val_outputs.cpu().numpy()
                        val_labels = val_labels.cpu().numpy()
                        for i in range(len(val_probs)):
                            plt.figure()

                            slice_index = int(val_positions[i,0,-1].item())
                            plt.subplot(2,2,1)
                            input = inputs[i,0,:,:,slice_index]
                            # plt.imshow(input)
                            plt.imshow(val_probs[i,0,:,:,slice_index], alpha=0.5)
                            plt.colorbar()
                            plt.title("0 - prediction")
                            plt.subplot(2,2,2)
                            # plt.imshow(input)
                            plt.imshow(val_labels[i,0,:,:,slice_index], alpha=0.5)
                            plt.title("0 - label")

                            plt.subplot(2,2,3)
                            slice_index = int(val_positions[i,1,-1].item())
                            input = inputs[i,0,:,:,slice_index]
                            # plt.imshow(input)
                            plt.imshow(val_probs[i,1,:,:,slice_index], alpha=0.5)
                            plt.colorbar()
                            plt.title("1 - prediction")
                            plt.subplot(2,2,4)
                            # plt.imshow(input)
                            plt.imshow(val_labels[i,1,:,:,slice_index], alpha=0.5)
                            plt.title("1 - label")

                            plt.savefig(os.path.join(os.path.dirname(__file__), "check_figures", f"epoch_{epoch}_batch_{b}_input_{i}"))
                            plt.show()

                    # check validation metric
                    check_metric = self.check_metric.aggregate().item()
                    self.check_metric.reset()
                    check_metric *= image_resolution[0]
                    val_metric_det = self.metric_det.aggregate().item()
                    self.metric_det.reset()
                    val_metric = val_metric_det
                    if self.metric_seg:
                        val_metric_seg = self.metric_seg.aggregate().item()
                        self.metric_seg.reset()
                        val_metric = [val_metric, val_metric_seg]
                    metric_values.append(val_metric)
                    
                    # update network if imporovement of validation metric
                    
                    if self.metric_seg:
                        val_metric = sum(val_metric)
                    if val_metric > best_metric:
                        best_metric = val_metric
                        best_metric_epoch = epoch + 1
                        torch.save(self.net_det.state_dict(), os.path.join(self.model_run_directory, "best_model.pth"))
                        
                    # weights and biases report
                    if wandb_key_directory:
                        metrics = {"train/train_det_loss": epoch_det_loss,
                                   "train/epoch": epoch + 1,
                                   "train/mEuclidDist(mm)": check_metric,
                                   "train/val_det_metric": val_metric_det,
                                   "train/learning_rate": self.learning_rate_scheduler.get_last_lr()[0]}
                        if self.loss_function_seg:
                            metrics.update({"train/train_seg_loss": epoch_seg_loss})
                        if self.metric_seg:
                            metrics.update({"train/val_seg_metric": val_metric_seg})
                        wandb.log(metrics)
                        
                    # report
                    print(f"Current average {' + '.join(validation_metric_name)} = {val_metric:.4f}")
                    print(f"Current average Euclidean distance = {round(check_metric, 3)} mm")
                    print(f"Best average {' + '.join(validation_metric_name)} = {best_metric:.4f}",
                          f"at epoch {best_metric_epoch}")
        
        # real training time
        print(f"\nTraining completed. Best {' + '.join(validation_metric_name)} = {best_metric:.4f}",
              f"at epoch {best_metric_epoch}")
        real_train_time_seconds = self.chronometer.stop()
        self.train_overview.add_general("real_train_time_seconds", real_train_time_seconds)
        real_train_time_hours = datetime.timedelta(seconds=real_train_time_seconds)
        print(f"Time for training : {real_train_time_hours} hours\n")
        
        # task prioritization hisotry
        # task_priority_history = self.task_prioritization.history()
        
        # check max GPU memory
        max_gpu_memory_GB = round(torch.cuda.max_memory_allocated(self.device)*1e-9,2)
        self.train_overview.add_general("max_gpu_memory_GB", max_gpu_memory_GB)
        
        # end training
        if wandb_key_directory:
            wandb.finish()
        self.train_overview.add_general("epoch_loss_values", epoch_loss_values)
        self.train_overview.add_general("metric_values", metric_values)
        self._plot_train()
        self._train_val_metric()
        self.train_overview.save()
        
        # set training done
        self.train_done = True

    # Load trained model
    def load_state(self, model_id_directory: str) -> None:
        
        # load train overview of model
        self.train_overview.load(model_id_directory)
        
        # get dataset
        num_processes = self.train_overview.get_general("num_processes")
        self._setup_data(self.train_overview.get_training(), num_processes=num_processes)
        
        # get model
        self._setup_model(self.train_overview.get_model())
        
        # load best model
        self.model_run_directory = model_id_directory
        self.net_det.load_state_dict(torch.load(os.path.join(self.model_run_directory, "best_model.pth")))
        
        # load inferer
        inference = self.train_overview.get_model("inference")
        if inference == "Simple":
            self.inferer = SimpleInferer()
        elif inference == "SlidingWindow":
            patch_size = self.train_overview.get_training("patch_size")
            self.inferer = SlidingWindowInferer(roi_size=patch_size, overlap=0.5)
        
        # load post processing
        post_processing_list = self.train_overview.get_model("post_processing")
        self.post_pred, _ = post_processing_transforms(post_processing_list)
        
        # load validation metric
        validation_metric = self.train_overview.get_training("validation_metric")\
        
        if validation_metric[0] == "DiceMetric":
            self.metric_det = DiceMetric()
        elif validation_metric[0] == "MeanIoU":
            self.metric_det = MeanIoU()
        elif validation_metric[0] == "MSEMetric":
            self.metric_det = MSEMetric()
        else:
            raise Exception(f"Non implemented validation metric '{validation_metric[0]}'")
        
        self.metric_seg = None
        if len(validation_metric)>1:
            if validation_metric[1] == "DiceMetric":
                self.metric_det = DiceMetric()
            else:
                raise Exception(f"Non implemented validation metric '{validation_metric[1]}'")
                
        self.check_metric = MSEMetric()
            
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
        
        inference_time = 0
        inference_memory = 0
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated(self.device)

        # evaluation
        with torch.no_grad():
            
            for i, test_batch_data in enumerate(self.test_loader):
                
                start_time = time.time()
                
                test_inputs, test_labels, test_positions = (test_batch_data["image"].to(self.device),
                                                            test_batch_data["label"].to(self.device),
                                                            test_inputs["label_lm_positions"],)

                # prediction on validation set
                test_outputs = self.net_det(test_inputs)
                if self.activation:
                    test_outputs = torch.sigmoid(test_outputs)
                test_centers = extract_landmarks_from_heatmaps(test_outputs)
                
                inference_time += (time.time() - start_time)
                inference_memory = (torch.cuda.max_memory_allocated(self.device) - initial_memory)/(1000**3)
                
                self.check_metric(test_centers, test_positions)
                if self.metric_seg:
                    self.metric_seg(test_outputs[:-1], test_labels[:-1])
                    self.metric_det(test_outputs[-1], test_labels[-1])
                else:
                    self.metric_det(test_outputs, test_labels)

                # save outputs
                for batch_number in range(len(test_outputs)):
                    
                    # plot output vs label
                    identifier = i*len(test_outputs)+batch_number+1
                    plt.figure("check", (12, 18))
                    num_channels = self.train_overview.get_model("out_channels")
                    for i in range(num_channels):
                        # move the data to the CPU
                        slice_number = int(test_positions.cpu()[batch_number,i,2].item())
                        test_images_slice = test_inputs.cpu()[batch_number, 0, :, :, slice_number]
                        test_labels_slice = test_labels.cpu()[batch_number, i, :, :, slice_number]
                        test_outputs_slice = test_outputs[batch_number].cpu()[i][:, :, slice_number]
                        plt.subplot(num_channels, 2, 2*i+1)
                        plt.imshow(test_images_slice, cmap="gray")
                        plt.imshow(np.where(test_labels_slice==0, np.nan, test_labels_slice))
                        plt.title(f"Label {identifier}")
                        plt.subplot(num_channels, 2, 2*i+2)
                        plt.imshow(test_images_slice, cmap="gray")
                        plt.imshow(np.where(test_outputs_slice<=0.01, np.nan, test_outputs_slice))
                        plt.title(f"Output {identifier} (channel {i})")
                        saver = SaveImage(output_dir=os.path.join(outputs_directory, str(identifier)), output_postfix=f"_{i+1}", separate_folder=False)
                        saver(test_outputs[batch_number][i])
                    plt.savefig(f"{figure_directory}/{identifier}.png")
                    plt.show()
                    

        # get validation metric on test dataset
        inference_time /= len(self.test_loader)
        check_metric = self.check_metric.aggregate().item()
        self.check_metric.reset()
        check_metric *= image_resolution[0]
        test_metric = self.metric_det.aggregate().item()
        self.metric_det.reset()
        if self.metric_seg:
            test_metric_seg = self.metric_seg.aggregate().item()
            self.metric_seg.reset()
            test_metric = [test_metric, test_metric_seg]
                
        print(f"\nInference time = {inference_time:.2f} s, GPU memory = {inference_memory} GB\n")
        if self.metric_seg:
            print(f"Testing {validation_metric_name[0]} = {test_metric[0]}, {validation_metric_name[1]} = {test_metric[1]}")
        else:
            print(f"Testing {validation_metric_name[0]} = {test_metric}")
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