import os
import numpy as np
import keras.applications as ka
import keras
from keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import clone_model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


wound_dict = {
    0: "Abrasions",
    1: "Bruises",
    2: "Burns",
    3: "Cut",
    4: "Ingrown nails",
    5: "Laceration",
    6: "Stab wound"
}

def load_model():
    '''
    Load in a model using the tf.keras.applications model and return it.
    Insert a more detailed description here
    '''
    # Base model from MobileNetV2
    # Include the imagenet weight (which includes the flowers classification)
    # Remove the top classifier
    # Make input image size to be (224, 224, 3)
    base_model = MobileNetV2(weights='imagenet', include_top = False, input_shape = (224, 224, 3)) 

    # Freeze the model, making all trainable weight to be un-trainable in preparation for transfer learning
    base_model.trainable = False
    
    # return the freezed model
    return base_model

# Data pre-processing
def load_data(path):
    '''
    Load in the dataset from its home path. Path should be a string of the path
    to the home directory the dataset is found in. Should return a numpy array
    with paired images and class labels.
    
    Insert a more detailed description here.
    '''

    image_size = (640, 640)  # Image size for MobileNetV2
    class_names = os.listdir(path)  # Lists all subdirectories which should be each flower folder
    class_names.sort()  # Ensure the class names are sorted for consistency

    # Declare lists for containing all images and labels
    images = []
    labels = []

    # Check if path/directory of images exist, then 
    for label, class_name in enumerate(class_names):
        # Path name 
        class_dir = os.path.join(path, class_name)

        # Defensive programming - check if the path exists
        if os.path.isdir(class_dir):

            # Iterate through each file
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)

                # Set only acceptable file to be image file
                if img_path.endswith(('jpg', 'jpeg', 'png')):

                    # Load the image, using tf.keras.preprocessing library
                    # Resize the image by giving a target_size
                    img = image.load_img(img_path)
                    size = (224, 224)
                    img = img.resize(size)

                    # Change the image type to be ndarray type
                    img_array = image.img_to_array(img)
                    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNetV2
                    images.append(img_array)
                    labels.append(label-1)

    # X is the list of images, Y is the list of labels
    X = np.array(images)
    Y = np.array(labels)

    return X, Y


def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    To see what type train, test, and eval should be, refer to the inputs of 
    transfer_learning().
    
    (!)train, test and eval should be a list or tuple of images and labels in the form (image,label),
    this is format returned by load_data
    
    
    """
    # Combine X and Y for easier shuffling and splitting
    data = list(zip(X, Y))
   
    # shuffle the data is 'randomize' is specified
    if randomize:   
       np.random.shuffle(data)
       
    # Calculate split indices
    total_samples = len(data)
    train_size = int(total_samples * train_fraction)
    test_size = total_samples - train_size

    # Split the data
    train_data = data[:train_size]

    # If validation is specified, spare a half of test set to be validation set 
    if eval_set:
        eval_size = test_size // 2
        test_data = data[train_size:train_size + eval_size]
        eval_data = data[train_size + eval_size:]
        # return the train set, test set, validation set
        return (np.array([x[0] for x in train_data]), np.array([x[1] for x in train_data])), \
               (np.array([x[0] for x in test_data]), np.array([x[1] for x in test_data])), \
               (np.array([x[0] for x in eval_data]), np.array([x[1] for x in eval_data]))
    else:
        test_data = data[train_size:]
        # return the train set, test set
        return (np.array([x[0] for x in train_data]), np.array([x[1] for x in train_data])), \
               (np.array([x[0] for x in test_data]), np.array([x[1] for x in test_data]))
    

def confusion_matrix(predictions, ground_truth, plot=False, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.

    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
        - plot: boolean. If true, create a plot of the confusion matrix with
                either matplotlib or with sklearn.
        - classes: a set of all unique classes that are expected in the dataset.
                   If None is provided we assume all relevant classes are in 
                   the ground_truth instead.
    Outputs:
        - cm: type np.ndarray of shape (c,c) where c is the number of unique  
              classes in the ground_truth
              
              Each row corresponds to a unique class in the ground truth and
              each column to a prediction of a unique class by a classifier
    '''

    # Calculate the number of classes, using np.unique to find distinguish element (if all_classes not specified)
    num_classes = len(all_classes) if all_classes is not None else len(np.unique(ground_truth))
    # Initialise a zero matrix, with size of number of classes x number of classes.  
    cm = np.zeros((num_classes, num_classes), dtype = int)
    
    #iterate through out the whole array 
    for gt, pred in zip(ground_truth, predictions):
        # Update the cm matrix, if the ground_truth and prediction match the row and column indices, +1.  
        cm[gt, pred] += 1
        
    if plot == True: 
        # Using sklearn confusion matrix module to display the confusion matrix
        cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_disp.plot()
        
    # return the confusion matrix (type = ndarray)    
    return cm 
    

def precision(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's precision
    
    Inputs: see confusion_matrix above
    Outputs:
        - precision: type np.ndarray of length c,
                     values are the precision for each class
    '''
    # Precision is (true positive) / (true positive + false positive)
    # np.diag is true positives, np.sum sums up each row or column (depends on axis input)
    cm = confusion_matrix(predictions, ground_truth)
    precision = np.diag(cm)/np.sum(cm, axis = 0)
    
    return precision

def recall(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's recall
    
    Inputs: see confusion_matrix above
    Outputs:
        - recall: type np.ndarray of length c,
                     values are the recall for each class
    '''
    # Recall is (true positive) / (true positive + false negative)
    # Same logic as precision just a different axis
    cm = confusion_matrix(predictions, ground_truth)
    recall = np.diag(cm)/np.sum(cm, axis = 1)
    return recall

def f1(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's f1 score
    Inputs:
        - see confusion_matrix above for predictions, ground_truth
    Outputs:
        - f1: type nd.ndarry of length c where c is the number of classes
    '''
    # F1 score formula is (2 * Precision * Recall) / (Precision + Recall)
    P = precision(predictions, ground_truth)
    R = recall(predictions, ground_truth)
    f1 = (2 * P * R)/(P + R)
    
    return f1


def transfer_learning(train_set, eval_set, test_set, model, parameters = (0.01, 0.9, False)):
    '''
    Implement and perform standard transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)
            
    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''

    # Find SGD parameters
    learning_rate, momentum, nesterov = parameters
    num_classes = 7
    
    # Create a new model on top of the base model from MobileNetV2 with the new layers 
    baymax = keras.Sequential([
        model,
        layers.GlobalAveragePooling2D(),  # Add global average pooling layer: converts maps from base model into single vector per image
        layers.Dense(num_classes, activation='softmax')  # Add a Dense layer with softmax activation
    ])
    
    # Compile model so it is useable
    B_optimizer = keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = nesterov)

    # Select optimiser to be SGD optimiser, metrics to be accuracy
    baymax.compile(optimizer =  B_optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ['accuracy'])
    
    # Extract images and labels from data sets generated in split data
    (train_images, train_labels) = train_set
    if eval_set is not None:
        (eval_images, eval_labels) = eval_set
    (test_images, test_labels) = test_set
    
    # Model training with epoch 10. 
    epochs = 10
    if eval_set is not None:    
        history = baymax.fit(train_images, train_labels, epochs = epochs , validation_data = (eval_images, eval_labels))
    else:
        history = baymax.fit(train_images, train_labels, epochs = epochs)    
    
    # Plot loss and accuracy against epoch, comparing the training data and
    # eval data to determine the performance of current model

    # Model Evalutation and Metric calculation
    predictions = baymax.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Performance analysis: Accuracy, confusion matrix, precision, recall, f1  
    cm = confusion_matrix(predicted_labels, test_labels, plot = True)
    accuracy = np.sum(predicted_labels == test_labels) / test_labels.size
    precision_values = precision(predicted_labels, test_labels)
    recall_values = recall(predicted_labels, test_labels)
    f1_values = f1(predicted_labels, test_labels)
    metrics = [accuracy, precision_values, recall_values, f1_values]
    
    return baymax, metrics

   
def accelerated_learning(train_set, eval_set, test_set, model, parameters = (0.01, 0.9, False)):
    '''
    Implement and perform accelerated transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)


    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''

    # Find SGD parameters
    learning_rate, momentum, nesterov = parameters
    num_classes = 7
    
    # Create the classification layer that takes in the shape of the extracted features.
    baymax = keras.Sequential([
        keras.Input(shape = (7, 7, 1280)), # From last layer of base_model
        layers.GlobalAveragePooling2D(),  # Add global average pooling layer: converts maps from base model into single vector per image
        layers.Dense(num_classes, activation='softmax')  # Add a Dense layer with softmax activation
    ])
    
    #Compile model so it is useable
    B_optimizer = keras.optimizers.SGD(learning_rate = learning_rate,momentum = momentum, nesterov = nesterov)

    baymax.compile(optimizer =  B_optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ['accuracy'])
    
    #Separate images and labels into their own variables
    (train_images, train_labels) = train_set
    (eval_images, eval_labels) = eval_set
    (test_images, test_labels) = test_set
        
    #Feature extract the images using base model then save into variable
    train_features_images = model.predict(train_images)
    eval_features_images = model.predict(eval_images)
    test_features_images = model.predict(test_images)

    
    #Model training
    epochs = 10
    if eval_set is not None:    
        history = baymax.fit(train_features_images, train_labels, epochs = epochs ,validation_data = (eval_features_images, eval_labels))
    else:
        history = baymax.fit(train_features_images, train_labels, epochs = epochs)    

    baymax.summary(show_trainable = True)
    
    #Plot loss and accuracy against epoch, comparing the training data and
    #eval data to determine the performance of current model

    #Model Evalutation
    predictions = baymax.predict(test_features_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(predicted_labels, test_labels, plot = True)
    accuracy = np.sum(predicted_labels == test_labels) / test_labels.size
    precision_values = precision(predicted_labels, test_labels)
    recall_values = recall(predicted_labels, test_labels)
    f1_values = f1(predicted_labels, test_labels)
    metrics = [accuracy, precision_values, recall_values, f1_values]
    
    return baymax, metrics


if __name__ == "__main__":
    base = load_model()
    X, Y = load_data('./image_data')
    train_set, test_set, eval_set = split_data(X, Y, 0.6, True, True)
    baymax, metrics = accelerated_learning(train_set, eval_set, test_set, base)
    print(metrics)