import wget, zipfile, os, logging

logging.basicConfig(level = logging.DEBUG)

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
            os.mkdir(self.directory_to)
        except:
            logging.info(self.directory_to+" already exists")

        for dir in ['yes', 'no']:
            try:
                os.mkdir(os.path.join(self.directory_to, dir))
            except:
                logging.info(os.path.join(self.directory_to, dir)+" already exists")

    def run(self):
        wget.download(self.url, self.directory_to + 'data.zip')
        logging.info('Zip téléchargé')
        with zipfile.ZipFile(self.directory_to + 'data.zip', 'r') as zip_ref:
            zip_ref.extractall(self.directory_to)
            logging.info('Dossier dézippé')
        #Au lieu de les mettre sur un dossier il faudra les mettre dans la BDD quand elle sera créée
        os.remove(self.directory_to + 'data.zip')        
            
