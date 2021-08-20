# Create datascience with audio transforms and save as .csv
import os
import librosa.display, librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat, loadmat

################################         SOME FUNCTION DEFINITIONS           ###########################################


def AudioArray(audio_path, samples):
    samplerate = librosa.get_samplerate(audio_path)
    duration = librosa.get_duration(librosa.load(audio_path, sr=samplerate, res_type='kaiser_fast')[0])
    return librosa.load(audio_path, sr=samples * samplerate / 22050 / duration, res_type='kaiser_best')[0:samples]


def MelSpectrogamArray(audio_path, samples, Mel_fmax):
    resampled_audio, new_samplerate = AudioArray(audio_path, samples)
    S = librosa.feature.melspectrogram(y=resampled_audio, sr=new_samplerate, n_mels=128, fmax=Mel_fmax)
    return librosa.power_to_db(S, ref=np.max)

def Normalized(X_array):
    return (X_array - X_array.min()) / (X_array.max() - X_array.min())


def CompileSpectrograms(spectrograms_path):
    dBvsTime_list = []
    for audio_path in spectrograms_path:
        for spectrogram_row in Normalized(MelSpectrogamArray(audio_path, samples, Mel_fmax)):
            dBvsTime_list.append(list(spectrogram_row))
    return dBvsTime_list

def ExtractClasses(audio_paths):
    return [audio_path.split('\\')[-1].split('_')[0] for audio_path in audio_paths]


def show_spectrogram(audio_path, samples, Mel_fmax):
    resampled_audio, new_samplerate = AudioArray(audio_path, samples)
    spectrogram_array = librosa.feature.melspectrogram(y=resampled_audio, sr=new_samplerate, n_mels=128, fmax=Mel_fmax)
    fig, ax = plt.subplots()
    S_dB_unit = Normalized(spectrogram_array)
    img = librosa.display.specshow(S_dB_unit, x_axis='time', y_axis='mel', sr=new_samplerate, fmax=Mel_fmax, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram : '+ '\n' + audio_path.split('\\')[-1])
    plt.show()


################################      AUDIO FILE PROCESSING AND STORAGE      ###########################################

relative_voice_path = "..\\..\\GitHub\\recognition_of_voice_commands_in_Spanish\\voices without noise"
file_names = os.listdir(relative_voice_path)
samples = 32000  # samples for audio array
Mel_fmax = 3000

###################################  AUDIO FILE PATHS FOR TRAINING  ####################################################
ENCENDER_voice_paths = [relative_voice_path + "\\" + file_name for file_name in file_names if "ENCENDER_" in file_name]
APAGAR_voice_paths = [relative_voice_path + "\\" + file_name for file_name in file_names if "APAGAR_" in file_name]
_0_voice_paths = [relative_voice_path + "\\" + file_name for file_name in file_names if "0_" in file_name and "50_" not in file_name and "100_" not in file_name]
_25_voice_paths = [relative_voice_path + "\\" + file_name for file_name in file_names if "25_" in file_name]
_50_voice_paths = [relative_voice_path + "\\" + file_name for file_name in file_names if "50_" in file_name]
_75_voice_paths = [relative_voice_path + "\\" + file_name for file_name in file_names if "75_" in file_name]
_100_voice_paths = [relative_voice_path + "\\" + file_name for file_name in file_names if "100_" in file_name]

############### Power vs. time at any frequency for any frequency in the human voice register  #########################
dBvsTime_ENCENDER   = CompileSpectrograms(ENCENDER_voice_paths)
dBvsTime_APAGAR     = CompileSpectrograms(APAGAR_voice_paths)
dBvsTime_0          = CompileSpectrograms(_0_voice_paths)
dBvsTime_25         = CompileSpectrograms(_25_voice_paths)
dBvsTime_50         = CompileSpectrograms(_50_voice_paths)
dBvsTime_75         = CompileSpectrograms(_75_voice_paths)
dBvsTime_100        = CompileSpectrograms(_100_voice_paths)

# BINARY CLASSIFICATION BETWEEN "ENCENDER" AND "APAGAR"
ON_OFF_classes = ['ENCENDER']*len(dBvsTime_ENCENDER) + ['APAGAR']*len(dBvsTime_APAGAR)

# CLASSIFICATION OF ALL VOICE COMMANDS
ALL_classes = ['ENCENDER']*len(dBvsTime_ENCENDER) + ['APAGAR']*len(dBvsTime_APAGAR) +  ['0']*len(dBvsTime_0) + \
              ['25']*len(dBvsTime_25) + ['50']*len(dBvsTime_50) + ['75']*len(dBvsTime_75) + ['100']*len(dBvsTime_100)

# Export to MATLAB file
savemat(".\\Data\\Voice_Command_dBvsTime.mat",
{
    "dBvsTime_ENCENDER"   :     dBvsTime_ENCENDER,
    "dBvsTime_APAGAR"     :     dBvsTime_APAGAR,
    "dBvsTime_0"          :     dBvsTime_0,
    "dBvsTime_25"         :     dBvsTime_25,
    "dBvsTime_50"         :     dBvsTime_50,
    "dBvsTime_75"         :     dBvsTime_75,
    "dBvsTime_100"        :     dBvsTime_100,
})
