import dcase_util

filename = "TAU dataset"
filepath = "/mnt/nas/home/chenjiafeng/"

dataset = dcase_util.datasets.TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet(storage_name=filename, data_path=filepath)

dataset.download_packages()