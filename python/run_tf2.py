import os
import sys
import shutil
import os.path, time
from PIL import Image
from datetime import datetime, timedelta
import numpy as np
import uuid

import train_test_functions as ft

"""
Three-fold cross-validation train/test
- Train/test each fold in a folder (pre-saved by standardise_images)
- report confusion
"""

# Global variable: persists the model
model = None

def confusion(answers, filename = None):
    '''
    Generates a confusion matrix from the returned predictions
    Input: class/prediction pairs
    TODO refactor and move to utils?
    '''

    if filename: f = open(filename, 'w')
  
    correct = 0
    incorrect = 0
    confusion = {}
    classes = []
    for answer in answers:
        # Make sure the answer and prediction are in dicts
        if answer[0] not in classes:
            classes.append(answer[0])
        if answer[1] not in classes:
            classes.append(answer[1])
        if answer[0] not in confusion:
            confusion[answer[0]] = {}
        actual = confusion[answer[0]]
        if (answer[1] not in actual):
            actual[answer[1]] = 0
     
        actual[answer[1]] += 1
        if answer[0] == answer[1]:
            correct += 1
        else:
            incorrect += 1
            print("WRONG:{} actual:{} predicted:{} score:{}".format(answer[3], answer[0], answer[1], answer[2]))

    # Now print it

    # Set up precision totals
    predicted_tp = {}
    predicted_fp = {}
    for cls in classes:
        predicted_tp[cls] = 0
        predicted_fp[cls] = 0

    # Output the header
    classes.sort()
    print('ACTUAL/PREDICTED'.ljust(30,' ') + '\t'.join(classes) + '\tRECALL')
    if filename: f.write('ACTUAL/PREDICTED'.ljust(25,' ') + '\t' + '\t'.join(classes) + '\tRECALL\n')

    # Output the rows
    for cls in classes:
        print(cls.ljust(30,' '), end = '\t')
        if filename: f.write(cls.ljust(30,' ') + '\t')
        tp = 0
        fn = 0
        # Output the columns
        for cls2 in classes:
            score = (cls in confusion and cls2 in confusion[cls] and confusion[cls][cls2]) or 0
            if cls == cls2:
                tp += score
                predicted_tp[cls2] += score
            else:
                fn += score
                predicted_fp[cls2] += score
            print(str(score), end = '\t')
            if filename: f.write(str(score) + '\t')
        # Output recall (if any)
        if (tp+fn) > 0:
          print('{:.3f}'.format(tp/(tp+fn)))
          if filename: f.write('{:.3f}'.format(tp/(tp+fn)) + '\n')
        else:
          print('N/A')
          if filename: f.write('N/A\n')
        

    # Output the precision
    print('PRECISION'.ljust(25,' '), end = '')
    if filename: f.write('PRECISION'.ljust(25,' '))

    for cls in classes:   
        precision = (predicted_tp[cls] > 0 or predicted_fp[cls] > 0) and (predicted_tp[cls]/(predicted_tp[cls] + predicted_fp[cls])) or 0.0
        print('\t' + '{:.3f}'.format(precision), end = '')
        if filename: f.write('\t' + '{:.3f}'.format(precision))
    print('\n')
    if filename: f.write('\n\n')   

    # Output the accuracy
    print('correct: ' + str(correct) + ' (' + '{:.2f}'.format( 100* correct/(correct+incorrect)) + '%) incorrect: ' + str(incorrect))
    if filename: f.write('correct: ' + str(correct) + ' (' + '{:.2f}'.format( 100* correct/(correct+incorrect)) + '%) incorrect: ' + str(incorrect) + '\n')

    # TODO F-score etc

    if filename: f.close()
    

def main():
    '''
    Trains and tests three different 2/3 - 1/3 folds of the tephritid wing labelled images
    '''
    
    DATASET_PATH = 'C:/Users/harmera/OneDrive - MWLR/repos/tephritidML/'
    model_dir = DATASET_PATH + 'models/'
    log_dir = DATASET_PATH + 'logs/'
    model_name = 'Xception'
    labels_path = DATASET_PATH + 'labels/labels3.csv'
    confusion_results = log_dir + 'trupanea_v3_full_results.txt'

    # for i in range(1,4):
    #     dataset_name = 'trupanea_V1_{}'.format(i)
    #     images_path = DATASET_PATH + 'img/trupanea_model/img_folds/{}/'.format(i)
    #     train_data_dir = images_path + 'train/'
    #     valid_data_dir = images_path + 'val/'
    #     ft.retrain(model_name, train_data_dir, valid_data_dir, model_dir, log_dir, dataset_name)
    
    results = []
    for i in range(1,4):
        model_file = model_dir + 'trupanea_V1_{}_Xception_transfer.h5'.format(i)
        images_path = DATASET_PATH + 'img/trupanea_model/img_folds/{}/'.format(i)
        test_data_dir = images_path + 'val/'
    
        _, answers = ft.test_model(model_file, labels_path, test_data_dir, model_name)
        results += answers
    
    confusion(results, confusion_results)
    
    # Train a full final model (without folds)
    # train_data_dir = 'C:/Users/harmera/OneDrive - MWLR/repos/tephritidML/img/trupanea_model/img_sorted'
    # dataset_name = 'trupanea_v2_full'
    # ft.retrain_final(model_name, train_data_dir, model_dir, log_dir, dataset_name)
    
    # Predict IDs of new images not used for training
    # model_path = model_dir + 'trupanea_v1_final_Xception_transfer.h5'
    # test_images_path = DATASET_PATH + 'img_unk_sort/'
    # preds = ft.predict_new(model_path, labels_path, test_images_path, model_name)
  
if __name__ == '__main__':
  main()
