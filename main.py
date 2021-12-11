from dataloader.data import FluentSpeechDATASET

if __name__ == '__main__':
    dataset = FluentSpeechDATASET(data_root='fluent_speech_commands_dataset', split='train')

    feat = dataset.__getitem__(2)['feats']
    print(feat.size())

    feat = feat.reshape(feat.shape[0], feat.shape[1]).permute(1,0)

    print(feat.size())

    # for i in range(0, 1000):
    #     temp = dataset.__getitem__(i)
    #     print(temp['feats_length'])

