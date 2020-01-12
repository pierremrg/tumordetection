import wget, zipfile, os, logging

logging.basicConfig(level = logging.INFO)

class PictureGetter:
    
    """
    Pour l'instant pas de paramètre car le zip a une adresse fixe sur le drive
    Rajout peut-être d'un futur paramètre pour identifier le hadoop distribué et pour stocker au bon endroit
    """
    def __init__(self, url, directory_to):
        logging.info('get_picture.init')
        self.url = url
        self.directory_to = directory_to

        self.createOutputDirectory()

    def createOutputDirectory(self):
        try:
            os.mkdir('tmp')
        except:
            logging.info("tmp already exists")

        for dir in ['yes', 'no']:
            try:
                os.mkdir(os.path.join('tmp', dir))
            except:
                logging.info(os.path.join('tmp', dir)+" already exists")

    def run(self):
        wget.download(self.url, 'tmp/data.zip')
        logging.info('Zip téléchargé')
        with zipfile.ZipFile('tmp/data.zip', 'r') as zip_ref:
            zip_ref.extractall('tmp/')
            logging.info('Dossier dézippé')
        #Au lieu de les mettre sur un dossier il faudra les mettre dans la BDD quand elle sera créée
        os.remove('tmp/data.zip')        
            
