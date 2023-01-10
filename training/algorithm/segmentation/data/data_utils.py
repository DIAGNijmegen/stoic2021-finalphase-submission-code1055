import nibabel as nib
import numpy as np
import os
import shutil
import scipy.ndimage as ndimage
import medpy.io as mpio

def create_resized_dataset(size=(256, 256, 256), name='resized256'):
    path_original = '/data/ssd1/kienzlda/data_covidsegmentation/COVID-19-20_v2/'
    path_original_train = os.path.join(path_original, 'Train')
    path_original_val = os.path.join(path_original, 'Validation')
    #path_resized = '/data/ssd1/kienzlda/data_covidsegmentation/' + name
    path_resized = '/data/ssd1/kienzlda/data_stoic/segmentation/' + name
    path_resized_train = os.path.join(path_resized, 'Train')
    path_resized_val = os.path.join(path_resized, 'Validation')
    os.makedirs(path_resized_train, exist_ok=True)
    os.makedirs(path_resized_val, exist_ok=True)

    for path_o, path_r in [(path_original_train, path_resized_train), (path_original_val, path_resized_val)]:
        for data_name in sorted(os.listdir(path_o)):
            print(data_name)
            source = nib.load(os.path.join(path_o, data_name))
            source = source.get_fdata()

            scale_factors = (size[0] / source.shape[-3], size[1] / source.shape[-2], size[2] / source.shape[-1])
            if data_name.strip('.nii.gz')[-2:] == 'ct':
                target = ndimage.zoom(source, scale_factors)
                target = np.clip(target, -1000., 500.)
                target = (target + 1000) / 1500
            else:
                target = ndimage.zoom(source, scale_factors, order=0)

            np.save(os.path.join(path_r, data_name).strip('.nii.gz') + '.npy', target)



def create_resized_unsupervised_dataset_stoic(size=(256, 256, 256), name='resized256'):
    path_original = '/data/ssd1/kienzlda/data_stoic/data/mha/'
    #path_resized = '/data/ssd1/kienzlda/data_covidsegmentation/' + name + '/unsupervised_stoic'
    path_resized = '/data/ssd1/kienzlda/data_stoic/segmentation/' + name + '/unsupervised_stoic'
    os.makedirs(path_resized, exist_ok=True)

    for data_name in sorted(os.listdir(path_original)):
        print(data_name)
        p = os.path.join(path_original, data_name)
        source = mpio.load(p)[0]
        scale_factors = (size[0] / source.shape[-3], size[1] / source.shape[-2], size[2] / source.shape[-1])
        target = ndimage.zoom(source, scale_factors)
        target = np.clip(target, -1000., 500.)
        target = (target + 1000) / 1500

        np.save(os.path.join(path_resized, data_name).strip('.mha') + '.npy', target)

    print('finished!')


def create_resized_unsupervised_dataset_tcia(size=(256, 256, 256), name='resized256'):
    path_original = '/data/ssd1/kienzlda/data_covidsegmentation/TCIA/set1'
    #path_resized = '/data/ssd1/kienzlda/data_covidsegmentation/' + name + '/unsupervised_tcia'
    path_resized = '/data/ssd1/kienzlda/data_stoic/segmentation/' + name + '/unsupervised_tcia'
    os.makedirs(path_resized, exist_ok=True)
    path_supervised = '/data/ssd1/kienzlda/data_covidsegmentation/COVID-19-20_v2/'
    path_supervised_train = os.path.join(path_supervised, 'Train')
    path_supervised_val = os.path.join(path_supervised, 'Validation')

    #save statistics of each image in supervised dataset in list_supervised
    list_supervised = []
    for path_s in [path_supervised_train, path_supervised_val]:
        for data_name in sorted(os.listdir(path_s)):
            source = nib.load(os.path.join(path_s, data_name))
            source = source.get_fdata()
            list_supervised.append((data_name, np.sum(source), source.shape))

    list_duplicates = []
    for data_name in sorted(os.listdir(path_original)):
        print(data_name)
        source = nib.load(os.path.join(path_original, data_name))
        source = source.get_fdata()
        su = np.sum(source)
        sh = np.shape
        is_supervised = False
        #test if data is already in supervised dataset
        for elem in list_supervised:
            if abs(elem[1] - np.sum(source)) < 10 and source.shape == elem[2]:
                is_supervised = True
                list_duplicates.append((elem[0], data_name))
                print('Duplicate!!!')
                break
        if is_supervised == False:
            scale_factors = (size[0] / source.shape[-3], size[1] / source.shape[-2], size[2] / source.shape[-1])
            target = ndimage.zoom(source, scale_factors)
            target = np.clip(target, -1000., 500.)
            target = (target + 1000) / 1500

            np.save(os.path.join(path_resized, data_name).strip('.nii.gz') + '.npy', target)

    print('number supervised data:', len(list_supervised))
    print('number duplicates:', len(list_duplicates))
    print('finished!')



def calc_stats_supervised():
    path_supervised = '/data/ssd1/kienzlda/data_covidsegmentation/resized/Train/'
    sum, sum_sq, num = 0, 0, 0
    for data_name in sorted(os.listdir(path_supervised)):
        if data_name.split('.')[-2][-2:] == 'ct':
            #print(data_name)
            p = os.path.join(path_supervised, data_name)
            img = np.load(p)
            num += img.shape[0] * img.shape[1] * img.shape[2]
            sum += img.sum()
            sum_sq += (img * img).sum()

    print('------')
    total_mean = sum / num
    total_var = (sum_sq / num) - (total_mean ** 2)
    total_std = np.sqrt(total_var)
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_unsupervised_stoic():
    path_unsupervised = '/data/ssd1/kienzlda/data_covidsegmentation/resized/unsupervised_stoic'
    sum, sum_sq, num = 0, 0, 0
    for data_name in sorted(os.listdir(path_unsupervised)):
        #print(data_name)
        p = os.path.join(path_unsupervised, data_name)
        img = np.load(p)
        num += img.shape[0] * img.shape[1] * img.shape[2]
        sum += img.sum()
        sum_sq += (img * img).sum()

    print('------')
    total_mean = sum / num
    total_var = (sum_sq / num) - (total_mean ** 2)
    total_std = np.sqrt(total_var)
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_unsupervised_tcia():
    path_unsupervised = '/data/ssd1/kienzlda/data_covidsegmentation/resized/unsupervised_tcia'
    sum, sum_sq, num = 0, 0, 0
    for data_name in sorted(os.listdir(path_unsupervised)):
        #print(data_name)
        p = os.path.join(path_unsupervised, data_name)
        img = np.load(p)
        num += img.shape[0] * img.shape[1] * img.shape[2]
        sum += img.sum()
        sum_sq += (img * img).sum()

    print('------')
    total_mean = sum / num
    total_var = (sum_sq / num) - (total_mean ** 2)
    total_std = np.sqrt(total_var)
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


def calc_stats_tcia():
    path_supervised = '/data/ssd1/kienzlda/data_covidsegmentation/resized/Train/'
    path_supervised2 = '/data/ssd1/kienzlda/data_covidsegmentation/resized/Validation/'
    path_unsupervised = '/data/ssd1/kienzlda/data_covidsegmentation/resized/unsupervised_tcia'
    sum, sum_sq, num = 0, 0, 0
    for path in [path_supervised, path_unsupervised]:#[path_supervised, path_supervised2, path_unsupervised]:
        for data_name in sorted(os.listdir(path)):
            if data_name.split('.')[-2][-2:] == 'ct' or path == path_unsupervised:
                # print(data_name)
                p = os.path.join(path, data_name)
                img = np.load(p)
                num += img.shape[0] * img.shape[1] * img.shape[2]
                sum += img.sum()
                sum_sq += (img * img).sum()

    print('------')
    total_mean = sum / num
    total_var = (sum_sq / num) - (total_mean ** 2)
    total_std = np.sqrt(total_var)
    print('mean:', total_mean)
    print('var:', total_var)
    print('std:', total_std)


if __name__ == '__main__':
    pass
    #create_resized_dataset(size=(128, 128, 64), name='resized128')
    #create_resized_unsupervised_dataset_tcia(size=(128, 128, 64), name='resized128')
    #create_resized_unsupervised_dataset_stoic(size=(128, 128, 64), name='resized128')
    #calc_stats_supervised()
    #calc_stats_unsupervised_tcia()
    #calc_stats_unsupervised_stoic()
    #calc_stats_tcia()



