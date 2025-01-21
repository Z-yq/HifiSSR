from models.hifigan import HiFiGANGenerator
from models.Unet import MaskMapping
import librosa
import torch
from torchaudio.transforms import Spectrogram
import soundfile  as sf
import argparse
import os

def main(args):
    sr=48000
    vocoder= HiFiGANGenerator(input_channels=256,upsample_rates=[5,4,4,3,2],upsample_kernel_sizes=[10,8,8,6,4],weight_norm=False,upsample_initial_channel=1024)

    ckpt=os.path.join(args.ckpt_path,'checkpoint.pt')

    ckpt = torch.load(ckpt,map_location='cpu')
    vocoder.load_state_dict(ckpt['voc_state_dict'])
    vocoder.eval()

    generator=MaskMapping(32,256)
    generator.load_state_dict(ckpt['unet_state_dict'])
    generator.eval()

    mel_front=Spectrogram(512,512,int(48000*0.01))
    ref_fp=args.ref_wav
    source_fp=args.source_wav
    out_fp=args.out_wav
    
    wav=librosa.load(source_fp,sr=sr)[0]
    source_mel=mel_front(torch.FloatTensor(wav).unsqueeze(0))[:,:-1]
    source_mel=torch.log10(source_mel+1e-6)
    source_mel=source_mel.unsqueeze(0)
    ref_wav=librosa.load(ref_fp,sr=sr)[0]
    ref_mel=mel_front(torch.FloatTensor(ref_wav).unsqueeze(0))[:,:-1]
    ref_mel=torch.log10(ref_mel+1e-6)
    with torch.no_grad():
        g_out=generator(source_mel,ref_mel)
        g_out_wav=vocoder(g_out)
        g_out_wav=g_out_wav.flatten()
    sf.write(out_fp,g_out_wav.cpu().data.numpy(),sr) 
    print(source_fp,'finished.....')
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./static', help='checkpoint path')
    parser.add_argument('--out_wav', type=str, default='./ssred.wav', help='out_wav path')
    parser.add_argument('--ref_wav', type=str, default='./static/p228_002.wav', help='ref_wav path')
    parser.add_argument('--source_wav', type=str, default='./static/syz.wav', help='source_wav path')
    args = parser.parse_args()
    main(args)