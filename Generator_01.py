import numpy as np
import pickle
import librosa
import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io




# Edited

# print("File      Path:", Path(__file__).absolute())
# print("Directory Path:", Path().absolute()) # Directory of



# img=mpimg.imread('data.png')
#
# lum_img = img[:, :, 0]
#
# # This is array slicing.  You can read more in the `Numpy tutorial
# # <https://numpy.org/doc/stable/user/quickstart.html>`_.
#
# plt.imshow(lum_img)
# plt.show()
#
# plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# plt.show()
# im=Image.open('data.png')
# px = im.load()
# print (px[4, 4])
# px[4, 4] = (0, 0, 0)
# print (px[4, 4])
# cordinate = x, y = 150, 59
#
# # using getpixel method
# print (im.getpixel(cordinate))
#
#
# # now plot it
# import matplotlib.pyplot as plt
# plt.plot(px)
# plt.ylabel('some numbers')
# plt.show()


def plot_2_vec(vec1,vec2,name,dir):
    plt.figure(figsize=(30,10))
    plt.plot(vec1.ravel()+1.30,'m',label='Original Wave')
    plt.plot(vec2.ravel(), 'c',label='New Wave')
    plt.show()

def plot_2_vec_save_plot(vec1,vec2,name,folder,title,step=0):
    plt.figure(figsize=(30,10))
    plt.title(f'{title}{step}', fontsize=20)
    plt.plot(vec1.ravel()+1.30,'m',label='Original Wave')
    plt.plot(vec2.ravel(), 'c',label='New Wave')
    plt.rcParams.update({'font.size': 22})
    plt.legend(framealpha=1, frameon=True);
    plt.savefig(os.path.join(folder, f'f{name}.png'))

percent_stretch=30 # min 2 max 100
percent_delite=30 # min 2 max 100
data_input="data_input"
data_export='data_export'
#pickle.dump(new_tretch, open(o) f)
def new_foldef(name,data_export):
    version = name
    folder = f'./{data_export}/{version}'
    try:
        os.mkdir(folder)
    except Exception:
        pass
    return folder

def calc_percent(whole,percent):
    number=whole/100
    number=number*percent
    #whole=whole-number
    return number

########### load data from wav files start
data,samplerate  = librosa.load('./pitch_shift1.wav') # wavfile.read('./data_input/pitch_shift1.wav')
#print(f"number of channels = {data[1].shape}")
# print('sampler ate = ',samplerate)
# length = data.shape/ samplerate
# print(f"length = {length}s")
# time = np.linspace(0., length, data.shape)
# plt.plot(time, data, label="Left channel")
# #plt.plot(time, data[: 1], label="Right channel")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()
########### load data from wav files end


########### load data from pickle files start
pickle_files=[]
names= sorted(os.listdir(data_input))
print('Names of pickle files in derectory ',data_input,' : ',names)
##get data
for file in names:
    pickle_files.append(pickle.load(open(os.path.join(data_input,file),'rb')))
########### augumentation generator

def normalization_D(In):
    min=np.min(In)
    max=np.max(In)
    normal=(In-min)/(max-min)
    return normal

# def wave_generator_compress(wave,step_delite):
#     size_data=len(wave)
#     new_data=np.copy(wave)
#     for i in range(size_data):
#         if i%step_delite == 0:
#             #new_data=np.delete(new_data,i)
#             new_data[i]=0
#         else:
#             pass
#     new_wave=new_data[new_data !=0]
#     wave_median = np.array(cv2.medianBlur(new_wave, 5))#median filter ker 5
#     ## visualization Median filter difer
#     wave_median = normalization_D(wave_median)
#     wave_median=wave_median.ravel()
#     wave= normalization_D(wave)
#     wave=wave.ravel()
#     new_wave = librosa.effects.pitch_shift(wave_median, sr=samplerate, n_steps=-30) #librosa.core.resample(wave_median, samplerate, 10000) # librosa.effects.pitch_shift(wave_median, sr=samplerate, n_steps=-6)
#     return new_wave
#
# def wave_generator_stretch(wave,step_stretch):
#     size_data=len(wave)
#     new_data=np.copy(wave)
#     for i in range(size_data-1):
#         if i%step_stretch == 0:
#             middle=new_data[i]-new_data[i+1]
#             middle=middle+new_data[i]
#             new_data=np.insert(new_data, i, middle)
#             #new_data=np.delete(new_data,i)
#             #new_data[i]=0
#         else:
#             pass
#     #new_wave=new_data[new_data !=0]
#     new_wave=new_data
#     wave_median = np.array(cv2.medianBlur(new_wave, 5))#median filter ker 5
#     ## visualization Median filter difer
#     wave_median = normalization_D(wave_median)
#     wave_median=wave_median.ravel()
#     wave= normalization_D(wave)
#     wave=wave.ravel()
#     original_wave_noSilence_shape = np.array(wave_median.shape)
#     start_tr = original_wave_noSilence_shape
#     new_wave = librosa.core.resample(wave_median, start_tr, original_wave_noSilence_shape - 2000)
#     #new_wave = librosa.effects.pitch_shift(wave_median, sr=samplerate, n_steps=-30) #librosa.core.resample(wave_median, samplerate, 10000) # librosa.effects.pitch_shift(wave_median, sr=samplerate, n_steps=-6)
#     return new_wave

def wave_Aug_Generator_01_compress(wave,step_delite):
    norm_original_wave=normalization_D(wave)
    original_wave_noSilence= cv2.medianBlur(norm_original_wave, 5)
    original_wave_noSilence=np.reshape(original_wave_noSilence,len(original_wave_noSilence))
    original_wave_noSilence=normalization_D(original_wave_noSilence)
    original_wave_noSilence_shape = np.array(original_wave_noSilence.shape)
    start_tr = original_wave_noSilence_shape
    conversion_wave_sf3 = librosa.core.resample(original_wave_noSilence, start_tr, original_wave_noSilence_shape - step_delite)
    return  conversion_wave_sf3

def wave_Aug_Generator_01_stretch(wave,step_stretch):
    norm_original_wave=normalization_D(wave)
    original_wave_noSilence= cv2.medianBlur(norm_original_wave, 5)
    original_wave_noSilence=np.reshape(original_wave_noSilence,len(original_wave_noSilence))
    original_wave_noSilence=normalization_D(original_wave_noSilence)
    original_wave_noSilence_shape = np.array(original_wave_noSilence.shape)
    start_tr = original_wave_noSilence_shape
    conversion_wave_sf3 = librosa.core.resample(original_wave_noSilence, start_tr, original_wave_noSilence_shape + step_stretch)
    return  conversion_wave_sf3
epoch=3
difer=400
for j in range(epoch):
    folder=new_foldef(j, data_export)
    for i in range(len(pickle_files)):
        number=i
        data=np.copy(pickle_files[i])
        d=len(data)
        step_delite=calc_percent(len(data),percent_delite)
        step_stretch=calc_percent(len(data),percent_stretch)

        step_stretch = step_stretch + j * difer
        step_delite = step_delite - +j * difer

        new_compress=wave_Aug_Generator_01_compress(1/data,step_delite)   #wave_Augmentation_Generator_01
        name = f'epoch{j}_new_compress{step_delite}_new_wave{number}'
        pickle.dump(new_compress, open(os.path.join(folder, f'f{name}.pickle'), "wb"))
        plot_2_vec_save_plot(data, new_compress,name,folder,'step_delite =',step_delite)
        data = np.copy(pickle_files[i])
        new_stretch=wave_Aug_Generator_01_stretch(data,step_stretch)
        data=normalization_D(data).ravel()
        name = f'epoch{j}_step_stretch{step_stretch}_new_wave{number}'  # with open(folder, "wb") as f:
        pickle.dump(new_stretch, open(os.path.join(folder, f'f{name}.pickle'), "wb"))
        plot_2_vec_save_plot(data, new_stretch,name,folder,'step_stretch =',step_stretch)
## check rez
# names= sorted(os.listdir(folder))
# for file in names:
#     pickle_files.append(pickle.load(open(os.path.join(folder,file),'rb')))
#
# plot_2_vec(pickle_files[4], pickle_files[15])

# for i in range(len(pickle_files)):
#     data=np.copy(pickle_files[i])
#     new_compress=wave_Augmentation_Generator_01(data)
#     plot_2_vec(data, new_compress)
#     data = np.copy(pickle_files[i])
#     new_compress=wave_generator_compress(data,step_delite)
#     plot_2_vec(data, new_compress)
#     data=np.copy(pickle_files[i])
#     new_tretch=wave_generator_stretch(data,step_stretch)
#     #data=normalization_D(data).ravel()
#     plot_2_vec(data, new_tretch)
#     number=1
#     name = f'behavior{step_delite}_new_wave{number}'
#     # with open(folder, "wb") as f:
#     pickle.dump(new_tretch, open(os.path.join(folder, f'f{name}.pickle'), "wb"))



########### dump data too pickle files start


#pickle.dump(new_wave,open(os.path(data_export),'wb'))

#test data from file pickle
#data = pickle.dump(new_wave)
########### dump data too pickle files end



