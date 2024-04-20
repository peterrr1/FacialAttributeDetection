import csv
import os

class MetricSaver():
    """
    A class used to save evaluation metrics to CSV files.


    Attributes
    ----------
    metrics_dir : str
        a string representing the directory where the CSV files will be saved
    path_to_metadata : str
        a string representing the path to the metadata CSV file
    path_to_class_wise_accuracy : str
        a string representing the path to the class-wise accuracy CSV file
    path_to_validation_scores : str
        a string representing the path to the validation scores CSV file

    Methods
    -------
    create_model_metadata_csv(filename, header):
        Creates a new CSV file for model metadata if it doesn't already exist.
    save_model_metadata(*args):
        Appends a row of model metadata to the metadata CSV file.
    create_class_wise_acc_csv(filename, *args):
        Creates a new CSV file for class-wise accuracy if it doesn't already exist.
    save_class_wise_accuracy(model_id, epoch, *args):
        Appends a row of class-wise accuracy to the class-wise accuracy CSV file.
    create_validation_scores_csv(filename, header):
        Creates a new CSV file for validation scores if it doesn't already exist.
    save_validation_scores(model_id, epoch, *args):
        Appends a row of validation scores to the validation scores CSV file.
    """

    def __init__(self, path):
        self.metrics_dir = path

        self.path_to_metadata = ""
        self.path_to_class_wise_accuracy = ""
        self.path_to_validation_scores = ""




    def create_model_metadata_csv(self, filename, header):
        self.path_to_metadata = os.path.join(self.metrics_dir, filename)
        
        if os.path.isfile(self.path_to_metadata) is not True:
            print(f'Creating {filename}')

            with open(self.path_to_metadata, mode='w') as model_info_file:
                model_info_writer = csv.writer(model_info_file, delimiter=',')
                model_info_writer.writerow(header)
        else:
            print(f'{filename}.csv already exists.')



    def save_model_metadata(self, *args):
        if os.path.isfile(self.path_to_metadata) is True:
            print('Saving model metadata...')

            with open(self.path_to_metadata, mode='a') as model_info_file:
                model_info_writer = csv.writer(model_info_file, delimiter=',')
                model_info_writer.writerow([*args])




    def create_class_wise_acc_csv(self, filename, *args):
        self.path_to_class_wise_accuracy = os.path.join(self.metrics_dir, filename)

        if os.path.isfile(self.path_to_class_wise_accuracy) is not True:
            print(f'Creating {filename}')

            with open(self.path_to_class_wise_accuracy, mode='w') as class_wise_accuracy_file:
                class_wise_accuracy_writer = csv.writer(class_wise_accuracy_file, delimiter=',')
                class_wise_accuracy_writer.writerow(['model_id', 'epoch', *args])
        else:
            print(f'{filename} already exists.')
    


    def save_class_wise_accuracy(self, model_id, epoch, *args):
        with open(self.path_to_class_wise_accuracy, mode='a') as class_wise_accuracy_file:
            print('Saving class-wise accuracy...')

            class_wise_accuracy_writer = csv.writer(class_wise_accuracy_file, delimiter=',')
            class_wise_accuracy_writer.writerow([model_id, epoch, *args])




    def create_validation_scores_csv(self, filename, header):
        self.path_to_validation_scores = os.path.join(self.metrics_dir, filename)

        if os.path.isfile(self.path_to_validation_scores) is not True:
            print(f'Creating {filename}')

            with open(self.path_to_validation_scores, mode='w') as validation_scores_file:
                validation_scores_writer = csv.writer(validation_scores_file, delimiter=',')
                validation_scores_writer.writerow(header)
        else:
            print(f'{filename} already exists.')



    def save_validation_scores(self, model_id, epoch, *args):
        if os.path.isfile(self.path_to_validation_scores) is True:
            print('Saving validation scores...')

            with open(self.path_to_validation_scores, mode='a') as validation_scores_file:
                validation_scores_writer = csv.writer(validation_scores_file, delimiter=',')
                validation_scores_writer.writerow([model_id, epoch, *args])
# Path: data_manager/manage_csv.py