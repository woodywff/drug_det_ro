SUPPORTED_IMG_FORMAT = ['jpg', 'bmp', 'png']
H5FILE = 'dataset.h5'

# DEBUG_FLAG = True
DEBUG_FLAG = False

BACK_RANDOM = 0
BACK_PURE = 1
BACK_IMG = 2
# You need to specify yours
BACKGROUND_SOURCE = '../drug_det_data_gen/data/background'
BACKGROUND_FOLDER = 'data/background/'

from tqdm import tqdm
# from tqdm.notebook import tqdm

INFER_TEMP_IMG = 'infer_temp.jpg'
