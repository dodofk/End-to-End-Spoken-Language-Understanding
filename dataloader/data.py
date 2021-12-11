import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchaudio
import os
from sklearn import preprocessing


## ref: https://github.com/alexa/alexa-end-to-end-slu/
class FluentSpeechDATASET(Dataset):

    def __init__(self, data_root, split='train'):
        assert split in ['train', 'test', 'valid'], 'Invalid split'

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(self.data_root, 'data/{}_data.csv'.format(split)))
        self.df['intent'] = self.df[['action', 'object', 'location']].apply('-'.join, axis=1)

        self.intent_encoder = preprocessing.LabelEncoder()
        self.intent_encoder.fit(self.df['intent'])

        self.df['intent_label'] = self.intent_encoder.transform(self.df['intent'])

        self.labels_set = set(self.df['intent_label'])
        self.labels2index = dict()

        for label in self.labels_set:
            idx = np.where(self.df['intent_label']==label)[0]
            self.labels2index[label] = idx

    def __len__(self):
        return len(self.df)

    def load_audio(self, idx):
        df_row = self.df.iloc[idx]
        filename = os.path.join(self.data_root, df_row['path'])
        waveform, sr = torchaudio.load(filename)
        fbank_feats = torchaudio.compliance.kaldi.mfcc(waveform=waveform, num_ceps=40, num_mel_bins=80)
        intent = df_row['intent_label']
        transcription = df_row['transcription']
        return {
            'feats': fbank_feats,
            'feats_length': fbank_feats.shape[0],
            'waveform': waveform,
            'waveform_length': waveform[0].shape[0],
            'intent': intent,
            'transcription': transcription,
        }

    def __getitem__(self, index):
        return self.load_audio(index)

    def labels_list(self):
        return self.intent_encoder.classes_
