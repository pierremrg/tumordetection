import os
import logging
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance


formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

class DataAugmentation():
    """
    @param directory_from : Folder sith 2 folders, yes and now, where there are the pictures
    @param directory_to : Folder to save the transformation result
    @param max_augmentation : number of pictures wanted
    @param coef_rotation : number of images rotated in percent of the max_augmentation
    """
    def __init__(self, directory_from, max_augmentation, coef_rotation = 0.7, directory_to = None):
        logging.info('data_augmentation.init')
        
        self.directory_from = directory_from
        
        if (directory_to != None):
            self.directory_to = directory_to
            self.createOutputDirectory()
            self.copy_file()
        else:
            self.directory_to = directory_from
            
        self.MAX_AUGMENTATION = max_augmentation
        self.COEF_ROTATION = coef_rotation

    def createOutputDirectory(self):
        try:
            os.mkdir(self.directory_to)
        except:
            logging.info(self.directory_to+" already exists")

        for dir in ['yes', 'no']:
            try:
                os.mkdir(os.path.join(self.directory_to, dir))
            except:
                logging.info(os.path.join(self.directory_to, dir)+" already exists")
    
    def copy_file(self):
        images = os.listdir(self.directory_from + 'yes')
        for name in images:
            img = Image.open(self.directory_from + 'yes/' + name)
            img_name = img.filename.split('/')[-1]
            img = img.convert('RGB')
            img.save(self.directory_to + 'yes/' + img_name, format='jpeg')
        
        images = os.listdir(self.directory_from + 'no')
        for name in images:
            img = Image.open(self.directory_from + 'no/' + name)
            img_name = img.filename.split('/')[-1]
            img = img.convert('RGB')
            img.save(self.directory_to + 'no/' + img_name, format='jpeg')
                
    """
    Check if it is needed to equilibrate the dataset
    """   
    def equilibrate(self):
        logging.info('data_augmentation.equilibrate')
        
        image_yes = os.listdir(self.directory_from + 'yes')
        image_no = os.listdir(self.directory_from + 'no')
        
        nb_yes = len(image_yes)
        nb_no = len(image_no)
        
        ratio = nb_yes/nb_no
        
        if ratio == 1:
            logging.info('No need of augmentation')
        elif ratio <= 1:
            nb_img_to_add = nb_no - nb_yes
            self.compute_equilibrate(image_yes, 'yes/', nb_img_to_add)
        elif ratio >= 1:
            nb_img_to_add = nb_yes - nb_no
            self.compute_equilibrate(image_no, 'no/', nb_img_to_add)        
        
    """
        Equilibrate the dataset
        @param img_sources : list of images used to equilibrate
        @param directory_to_add : directory where the images are added
        @nb_img_to_add : number of images to add
    """
    def compute_equilibrate(self, img_sources, directory_to_add, nb_img_to_add):
        logging.info('data_augmentation.compute_equilibrate')
        logging.info('Adding %i images to %s', nb_img_to_add, self.directory_to + directory_to_add)
        
        if nb_img_to_add > len(img_sources):
            logging.info('Not enough images in source, adding %i images', len(img_sources))
            nb_img_to_add = len(img_sources)
        
        img_to_add = img_sources[:nb_img_to_add]
        
        logging.info('data_augmentation.compute_flip')
        new_images = list(map(lambda img : self.compute_flip(img, directory_to_add), img_to_add))

    """
    @param img_name : name of the image used 
    @param directory : directory where the images are added
    """    
    def compute_flip(self, img_name, directory):        
        img = Image.open(self.directory_from + directory + img_name)
        new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_name = os.path.splitext(img.filename)[0].split('/')[-1]
        new_name = img_name + '_flip.jpg'
        new_img = new_img.convert("RGB")
        new_img.save(self.directory_to + directory + new_name, format='jpeg')
    
        return new_name   
    
    def augmentation(self):
        logging.info('data_augmentation.augmentation')
        self.perform_rotate()
        self.apply_filters()       
    
    def perform_rotate(self):
        logging.info('data_augmentation.perform_rotate')
        directory_yes = os.listdir(self.directory_to + 'yes')
        directory_no = os.listdir(self.directory_to + 'no')
        
        nb_yes = len(directory_yes)
        nb_no = len(directory_no)
        
         # augmentation des yes
        images_yes = os.listdir(self.directory_to + 'yes')   
        nb_images_yes_to_add = self.MAX_AUGMENTATION * self.COEF_ROTATION - nb_yes
        nb_rotations = int(nb_images_yes_to_add/nb_yes)+1
        
        current_image_rotated = 0
        while nb_images_yes_to_add > 0:
            img = Image.open(self.directory_to + "/yes/" + images_yes[current_image_rotated])
            filenameToSave = os.path.splitext(img.filename)[0]
            img = img.convert("RGB")
            for i in range(nb_rotations):
                dg = np.random.randint(-45, 45, 1)
                new_img = img.rotate(dg, expand=True)
                new_name = filenameToSave + '_rotate' + str(dg) + ".jpg"
                try:
                    new_img.save(new_name, format='jpeg')
                    nb_images_yes_to_add -= 1

                except IOError :
                    logging.info('perform_rotate.save_failed')
            
            current_image_rotated += 1
            
         # augmentation des no
        images_no = os.listdir(self.directory_to + 'no')   
        nb_images_no_to_add = self.MAX_AUGMENTATION * self.COEF_ROTATION - nb_no
        nb_rotations = int(nb_images_no_to_add/nb_no)+1
        
        current_image_rotated = 0
        while nb_images_no_to_add > 0:
            img = Image.open(self.directory_to + "/no/" + images_no[current_image_rotated])
            filenameToSave = os.path.splitext(img.filename)[0]
            img = img.convert("RGB")

            for i in range(nb_rotations):
                dg = np.random.randint(-45, 45, 1)
                new_img = img.rotate(dg, expand=True)
                new_name = filenameToSave + '_rotate' + str(dg) + ".jpg"
                try:
                    new_img.save(new_name, format='jpeg')
                    nb_images_no_to_add -= 1
                except IOError :
                    logging.info('perform_rotate.save_failed')
                

            current_image_rotated += 1    
    
    def apply_filters(self):
        logging.info('data_augmentation.apply_filters')
        directory_yes = os.listdir(self.directory_to + 'yes')
        directory_no = os.listdir(self.directory_to + 'no')
        
        nb_yes = len(directory_yes)
        nb_no = len(directory_no)
        
        # augmentation des yes
        images_yes = os.listdir(self.directory_to + 'yes')   
        nb_images_yes_to_add = self.MAX_AUGMENTATION - nb_yes
        
        
        while nb_images_yes_to_add > 0:
            
            img = Image.open(self.directory_to + "/yes/" + images_yes[np.random.randint(0, nb_yes-1)])
            filenameToSave = os.path.splitext(img.filename)[0]
            img = img.convert("RGB")
            filterToApply = np.random.randint(0, 3)
            
            if filterToApply == 0:
                # appliquer mirrorTB
                new_name = filenameToSave + "_mirrorTB.jpg"
                new_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                
            elif filterToApply == 1 :
                # appliquer transpose
                new_name = filenameToSave + "_transpose.jpg"
                new_img = img.transpose(Image.TRANSPOSE)
                
            elif filterToApply == 2:
                # appliquer blur
                new_name = filenameToSave + "blur.jpg"
                new_img = img.filter(ImageFilter.BLUR)
                
            elif filterToApply == 3:
                # appliquer contraste
                appliedContrast = np.random.randint(0, 50, 1)
                new_name = filenameToSave + "_contrast" + str(appliedContrast) + ".jpg"

                enhancer = ImageEnhance.Sharpness(img)
                new_img = enhancer.enhance(appliedContrast)
                
            else:  
                print("other")
                
            try:
                new_img.save(new_name, format='jpeg')
                nb_images_yes_to_add -= 1                
            except IOError :
                logging.info('apply_filters.save_failed')
              
        # augmentation des no
        images_no = os.listdir(self.directory_to + 'no')   
        nb_images_no_to_add = self.MAX_AUGMENTATION - nb_no
        
        
        while nb_images_no_to_add > 0:
            img = Image.open(self.directory_to + "/no/" + images_no[np.random.randint(0, nb_no-1)])
            filenameToSave = os.path.splitext(img.filename)[0]
            img = img.convert("RGB")
            filterToApply = np.random.randint(0, 3)
            
            if filterToApply == 0:
                # appliquer mirrorTB
                new_name = filenameToSave + "_mirrorTB.jpg"
                new_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                
            elif filterToApply == 1 :
                # appliquer transpose
                new_name = filenameToSave + "_transpose.jpg"
                new_img = img.transpose(Image.TRANSPOSE)
                
            elif filterToApply == 2:
                # appliquer blur
                new_name = filenameToSave + "blur.jpg"
                new_img = img.filter(ImageFilter.BLUR)
                
            elif filterToApply == 3:
                # appliquer contraste
                appliedContrast = np.random.randint(0, 50, 1)
                new_name = filenameToSave + "_contrast" + str(appliedContrast) + ".jpg"

                enhancer = ImageEnhance.Sharpness(img)
                new_img = enhancer.enhance(appliedContrast)
                
            else:  
                print("other")
                
            try:
                new_img.save(new_name, format='jpeg')
                nb_images_no_to_add -= 1                
            except IOError :
                logging.info('apply_filters.save_failed')
    
    def run(self):
        self.equilibrate()
        self.augmentation()