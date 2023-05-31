'''
Script to crop, shrink and reorganise images for the wasps domain
'''

"""
- greyscale images and squish to be square
- organise images into class folders based on the spreadsheet
  + full name (species)
  + genus
  + subfamily  
"""


from PIL import Image, ImageDraw, ImageOps, ImageFilter
import os, shutil
from glob import glob
import numpy as np

def standardise_image(in_path, out_path, size=299):
    """
    Prepares the images for training/testing:
    - greyscale
    - resize (changes aspect). Note: assumes all are the same size, which they aren't! They have been cropped.
    """
    img = Image.open(in_path, mode='r')

    # Denoise
    # img = img.filter(ImageFilter.UnsharpMask) # POOR
    img = img.filter(ImageFilter.BLUR) # Works well
    
    # Grey scale
    img = np.array(ImageOps.grayscale(img)).astype(float)
    
    """
    # Standardise to 0..255 - NOT USED (doesn't alter the brightness)
    min = np.min(img)
    max = np.max(img)
    # print("min={} max={}".format(min, max))
    img = img * (255.0 / (max-min)) # Adjust contrast
    # print("New min={} max={}".format(np.min(img), np.max(img)))
    img = img - np.min(img)
    # print("Final min={} max={}".format(np.min(img), np.max(img)))
    """
    
    # Normalise to +- 2SD
    print("OLD mean={} min={} max={}".format(np.mean(img), np.min(img), np.max(img)))
    vmin = int(np.mean(img) - (2 * np.std(img)))
    vmax = int(np.mean(img) + (2 * np.std(img)))
    img = np.clip(img, vmin, vmax)
    img -= vmin
    img *= 256. / (vmax - vmin)

    print("NEW mean={} min={} max={}".format(np.mean(img), np.min(img), np.max(img)))
    
    """
    mean = np.mean(img)
    print("mean={}".format(mean))
    img = img + (127 - mean) # Adjust brightness
    print("New mean={}".format(np.mean(img)))
    """
    
    img = Image.fromarray(img.astype('uint8'),'L')
    
    # Resize - changes the aspect ratio
    img = img.resize((size, size))
    # img = img.convert('RGB')
    
    img.save(out_path)


def standardise_images(in_path, out_path, size=299):
    """
    Prepares the images for training/testing
    """
    for i, filename in enumerate(os.listdir(in_path)):
        filename_raw, ext = os.path.splitext(filename)
        print("'{}'".format(ext))
        if ext.lower() in ['.jpg', '.tif', '.png', '.bmp']:
            in_file_path = os.path.join(in_path, filename)
            out_file_path = os.path.join(out_path, filename_raw + '.png')
            print("{}: converting {} => {}".format(i+1, in_file_path, out_file_path))
            standardise_image(in_file_path, out_file_path, size)


def reorg_folders(in_path, out_path, index_file, label_name):
    '''
    - Get the list of files and their class from the spreadsheet
    - For each image:
        - filename = out_path + class/filename
        - create class folder if missing
        - copy the file
    '''
    
    # spreadsheet columns
    COLUMNS = ['file', 'country', 'genus', 'species' , 'fullName', 'fullName2', 'fullName3', 'accessionNumber', 'view', 'project']
    filename_col = 0
    label_col = COLUMNS.index(label_name)
    
    # Read the csv
    with open(index_file, 'r') as ifile:
        index = [line.split(',')  for line in ifile.readlines()]

    for i,item in enumerate(index[1:]):
        filename = item[filename_col]
        filename_raw, ext = os.path.splitext(filename)
        filename = filename_raw + '.png'
        label = item[label_col].strip().lower().replace(' ', '_')
        
        print("{}: '{}' = '{}'".format(i, filename, label))

        in_file_path = os.path.join(in_path, filename)
        out_file_dir = os.path.join(out_path, label)
        
        # Create class folder if needed
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)
   
        # Copy image to class folder
        shutil.copy(in_file_path, out_file_dir)
    print('ALL DONE, ITS A WRAP.')
    

def create_fold(in_path, out_path, folds, fold=1):
    """
    Copies the images to a train/test fold split.
    Uses round-robin in each subfolder - stratified folds.
    """
    
    folders = sorted(os.listdir(in_path))
    for folder in folders:
        print("=== Processing {} ===".format(folder))
        in_dir = os.path.join(in_path, folder)
        out_train = os.path.join(out_path, 'train', folder)
        out_val = os.path.join(out_path, 'val', folder)
        if not os.path.exists(out_train):
            os.makedirs(out_train)
        if not os.path.exists(out_val): 
            os.makedirs(out_val)
        
        files = sorted(os.listdir(in_dir))
        for i, filename in enumerate(files):
            in_file = os.path.join(in_dir, filename)
            if i % folds == (fold-1):
                print("{} VAL: {}".format(i+1, in_file))
                shutil.copy(in_file, out_val)
            else:
                print("{} TRAIN: {}".format(i+1, in_file))
                shutil.copy(in_file, out_train)

  
def main():
    
    # Test the image standardiser
    # IN_PATH = "D:/Dropbox/data/tephritidae/images_raw/NZspecies/NZAC04190927_Urophora_cardui_wing_dorsal_right.jpg"
    # OUT_PATH = "D:/Dropbox/data/tephritidae/images_standardised/NZAC04190927_test.png"
    # standardise_image(IN_PATH, OUT_PATH, size=299)
    
    # Standardise all images
    # IN_PATH = "C:/Users/harmera/OneDrive - MWLR/repos/tephritidML/img/img_raw"
    # OUT_PATH = "C:/Users/HarmerA/OneDrive - MWLR/repos/tephritidML/img/bactrocera_model/img_standardised"
    # standardise_images(IN_PATH, OUT_PATH, size=299)

    # Reorganise the images into class folders
    # IN_PATH = "C:/Users/HarmerA/OneDrive - MWLR/repos/tephritidML/img/trupanea_model/img_standardised"
    # OUT_PATH = "C:/Users/harmera/OneDrive - MWLR/repos/tephritidML/img/trupanea_model/img_sorted"
    # INDEX_FILE = "C:/Users/harmera/OneDrive - MWLR/repos/tephritidML/labels/fruitfly_annotationfile.csv"
    # LABEL_NAME = 'fullName3'
    # reorg_folders(IN_PATH, OUT_PATH, INDEX_FILE, LABEL_NAME)
    
    # # Create a 2/3 - 1/3 fold
    IN_PATH = 'C:/Users/harmera/OneDrive - MWLR/repos/tephritidML/img/trupanea_model/img_sorted'
    OUT_PATH = 'C:/Users/harmera/OneDrive - MWLR/repos/tephritidML/img/trupanea_model/img_folds/3'
    FOLDS = 3
    FOLD = 3
    create_fold(IN_PATH, OUT_PATH, FOLDS, FOLD)    
    
    
if (__name__ == '__main__'):
    main()
    