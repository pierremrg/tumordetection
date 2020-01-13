import os
import logging
import shutil
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from hdfs import InsecureClient


formatter = logging.Formatter(fmt = '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

hdfs_client = InsecureClient('http://192.168.1.4:9870', user='hadoop')

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
        self.directory_to = directory_to
        self.copy_file()
        
        self.MAX_AUGMENTATION = max_augmentation
        self.COEF_ROTATION = coef_rotation
    
    def copy_file(self):
        logging.info('data_augmentation.copy_file')
        hdfs_client.download('/' + self.directory_from, '')
        os.remove('data/data.csv')
        hdfs_client.upload('/' + self.directory_to, self.directory_from)
        shutil.rmtree(self.directory_from)	
                
    """
    Check if it is needed to equilibrate the dataset
    """   
    def equilibrate(self):
        logging.info('data_augmentation.equilibrate')
        
        image_yes = hdfs_client.list('/' + self.directory_from + 'yes')
        image_no = hdfs_client.list('/' + self.directory_from + 'no')
        
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
        with hdfs_client.read('/' + self.directory_from + directory + img_name) as reader:        
            img = Image.open(reader)

        new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_name = img_name.split('.')[0]
        new_name = img_name + '_flip.jpg'
        new_img = new_img.convert("RGB")

        with hdfs_client.write('/' + self.directory_to + directory + new_name) as writer:
            new_img.save(writer, format='jpeg')
    
        return new_name   
    
    def augmentation(self):
        logging.info('data_augmentation.augmentation')
        self.perform_rotate()
        self.apply_filters()       
    
    def perform_rotate(self):
        logging.info('data_augmentation.perform_rotate')
        directory_yes = hdfs_client.list('/' + self.directory_from + 'yes')
        directory_no = hdfs_client.list('/' + self.directory_from + 'no')
        
        nb_yes = len(directory_yes)
        nb_no = len(directory_no)
        
         # augmentation des yes
        images_yes = hdfs_client.list('/' + self.directory_from + 'yes')
        nb_images_yes_to_add = self.MAX_AUGMENTATION * self.COEF_ROTATION - nb_yes
        nb_rotations = int(nb_images_yes_to_add/nb_yes)+1
        
        current_image_rotated = 0
        while nb_images_yes_to_add > 0:
            img_name = images_yes[current_image_rotated]

            with hdfs_client.read('/' + self.directory_to + 'yes/' + img_name) as reader:
                img = Image.open(reader)

            filenameToSave = img_name.split('.')[0]
            img = img.convert("RGB")

            previous_dg = [0]
            for i in range(nb_rotations):
                dg = np.random.randint(-45, 45, 1)
                while(dg in previous_dg):
                    dg = np.random.randint(-45, 45, 1)
                previous_dg.append(dg)

                new_img = img.rotate(dg, expand=True)
                new_name = filenameToSave + '_rotate' + str(dg) + ".jpg"

                try:
                    with hdfs_client.write('/' + self.directory_to + 'yes/' +  new_name) as writer:
                        new_img.save(writer, format='jpeg')
                    nb_images_yes_to_add -= 1
                except IOError :
                    logging.info('perform_rotate.save_failed')
            
            current_image_rotated += 1
            
         # augmentation des no
        images_no = hdfs_client.list('/'+ self.directory_to + 'no')   
        nb_images_no_to_add = self.MAX_AUGMENTATION * self.COEF_ROTATION - nb_no
        nb_rotations = int(nb_images_no_to_add/nb_no)+1
        
        current_image_rotated = 0
        while nb_images_no_to_add > 0:
            img_name = images_no[current_image_rotated]

            with hdfs_client.read('/' + self.directory_to + 'no/' + img_name) as reader:
                img = Image.open(reader)

            filenameToSave = img_name.split('.')[0]
            img = img.convert("RGB")

            previous_dg = []
            for i in range(nb_rotations):
                dg = np.random.randint(-45, 45, 1)
                while(dg in previous_dg):
                    dg = np.random.randint(-45, 45, 1)
                previous_dg.append(dg)

                new_img = img.rotate(dg, expand=True)
                new_name = filenameToSave + '_rotate' + str(dg) + ".jpg"

                try:
                    with hdfs_client.write('/' + self.directory_to + 'no/' +  new_name) as writer:
                        new_img.save(writer, format='jpeg')
                    nb_images_no_to_add -= 1
                except IOError :
                    logging.info('perform_rotate.save_failed')                

            current_image_rotated += 1    
    
    def apply_filters(self):
        logging.info('data_augmentation.apply_filters')

        directory_yes = hdfs_client.list('/' + self.directory_to + 'yes')
        directory_no = hdfs_client.list('/' + self.directory_to + 'no')
        
        nb_yes = len(directory_yes)
        nb_no = len(directory_no)
        
        # augmentation des yes
        images_yes = hdfs_client.list('/' + self.directory_to + 'yes') 
        dict_yes = dict((name, []) for name in images_yes)  
        nb_images_yes_to_add = self.MAX_AUGMENTATION - nb_yes        
        
        while nb_images_yes_to_add > 0:
            img_name = images_yes[np.random.randint(0, len(images_yes)-1)]

            with hdfs_client.read('/' + self.directory_to + 'yes/' + img_name) as reader:            
                img = Image.open(reader)

            filenameToSave = img_name.split('.')[0]
            img = img.convert("RGB")

            filterToApply = np.random.randint(0, 3)
            while (filterToApply in dict_yes[img_name]):
                filterToApply = np.random.randint(0, 3)
            dict_yes[img_name].append(filterToApply)
            if len(dict_yes[img_name]) == 3:
                images_yes.remove(img_name)
          
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
                with hdfs_client.write('/' + self.directory_to + 'yes/' +  new_name) as writer:
                        new_img.save(writer, format='jpeg')
                nb_images_yes_to_add -= 1                
            except IOError :
                logging.info('apply_filters.save_failed')
              
        # augmentation des no
        images_no = hdfs_client.list('/' + self.directory_to + 'no')
        dict_no = dict((name, []) for name in images_no)   
        nb_images_no_to_add = self.MAX_AUGMENTATION - nb_no        
        
        while nb_images_no_to_add > 0:
            img_name = images_no[np.random.randint(0, len(images_no)-1)]

            with hdfs_client.read('/' + self.directory_to + 'no/' + img_name) as reader:  
                img = Image.open(reader)

            filenameToSave = img_name.split('.')[0]
            img = img.convert("RGB")

            filterToApply = np.random.randint(0, 3)
            while (filterToApply in dict_no[img_name]):
                filterToApply = np.random.randint(0, 3)
            dict_no[img_name].append(filterToApply)
            if len(dict_no[img_name]) == 3:
                images_no.remove(img_name)
            
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
                with hdfs_client.write('/' + self.directory_to + 'no/' +  new_name) as writer:
                        new_img.save(writer, format='jpeg')
                nb_images_no_to_add -= 1                
            except IOError :
                logging.info('apply_filters.save_failed')
    
    def run(self):
        self.equilibrate()
        self.augmentation()
