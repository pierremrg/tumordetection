from hdfs import InsecureClient

hdfs_cli = InsecureClient('http://192.168.1.4:9870', user='hadoop')

#hdfs_cli.delete('/images', recursive=True)
#hdfs_cli.delete('/images_augmented', recursive=True)
#hdfs_cli.delete('/images_crop', recursive=True)
#hdfs_cli.delete('/images_norm', recursive=True)
hdfs_cli.delete('/image_test', recursive=True)
hdfs_cli.delete('/image_test_crop', recursive=True)
hdfs_cli.delete('/image_test_ready', recursive=True)

#hdfs_cli.delete('/algo_trained', recursive=True)

