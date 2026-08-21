""" IMPORTS """

import os


""" GET ALL PATIENTS DISEASE CLASSIFICATION """

# Get all images and segmentations in one array with corresponding classification of disease in corresponing array
def get_all_patient_files(dataset_directory: str, report: bool = True) -> tuple[list[str], list[str], list[str]]:
    
    image_files = []
    segmentation_files = []
    landmark_files = []

    for root, dirs, files in os.walk(dataset_directory):
        for file in files:

            file_id = file.split(".")
            ext = '.'.join(file_id[1:])
            file_id = file_id[0]

            if not file_id.endswith("_image"):
                continue

            image_file = os.path.join(root, file)

            dir, train_val_test = os.path.split(root)
            train_val_test = train_val_test.replace("Images", "Labels")
            file_id = file_id.replace("_image", "") 
            segmentation_file = os.path.join(dir, train_val_test, "segmentation", f"{file_id}_label.{ext}")
            landmark_file = os.path.join(dir, train_val_test, "landmark", f"{file_id}_landmark.json")

            if not os.path.isfile(segmentation_file):
                for extenstion in ["bmp", "png" , "jpg"]:
                    segmentation_file = os.path.join(dir, train_val_test, "segmentation", f"{file_id}_label.{extenstion}")
                    if os.path.isfile(segmentation_file):
                        break
                    if extenstion == "jpg":
                        raise Exception(f"Segmentation file does not exist: {segmentation_file}")
            
            if not os.path.isfile(landmark_file):
                raise Exception(f"Landmark file does not exist: {landmark_file}")
            
            image_files.append(image_file)
            segmentation_files.append(segmentation_file)
            landmark_files.append(landmark_file)

    
    if len(image_files) != len(segmentation_files):
        raise Exception(f"Missing files, got {len(image_files)} and {len(segmentation_files)} segmentations.")
    
    if report:
        num_total = len(image_files)
        print("__"*30)
        print(f"Total image count: {num_total}\n")

    return image_files, segmentation_files, landmark_files


""" RUN """

if __name__ == '__main__':
    
    dataset_directory = ""
    image_files, segmentation_files, landmark_files = get_all_patient_files(dataset_directory)