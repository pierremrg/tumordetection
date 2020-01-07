import wget, zipfile, os, logging

logging.basicConfig(level = logging.DEBUG)

class PictureGetter:
    
    """
    Pour l'instant pas de paramètre car le zip a une adresse fixe sur le drive
    Rajout peut-être d'un futur paramètre pour identifier le hadoop distribué et pour stocker au bon endroit
    """
    def __init__(self):
        logging.info('get_picture.init')

    def run(self):
        url = 'https://docs.google.com/uc?export=download&id=1J1HAp--v1XF4qkEwkS0lrLG49Jic2HaF'
        wget.download(url, '../../')
        logging.info('Zip téléchargé')
        with zipfile.ZipFile('../../irm_images.zip', 'r') as zip_ref:
            zip_ref.extractall('../../')
            logging.info('Dossier dézippé')
        #Au lieu de les mettre sur un dossier il faudra les mettre dans la BDD quand elle sera créée
        os.remove("../../irm_images.zip")
        
            
