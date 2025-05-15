# Adapted from BigVGAN inference_e2e.py and inference_test_on_gamadhani_data.ipynb
# python inference_e2e_pretrained.py --output_dir /home/mila/k/krishna-maneesha.dendukuri/scratch/gamadhani/first_pass_experiments/adanorm_unprimed_32s_tf_v2/output
import os
import numpy as np
import argparse
import torch
from scipy.io.wavfile import write
import bigvgan

def inference(args):
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load the pretrained model from Hugging Face model hub
    print(f"Loading pretrained model '{args.pretrained_model}'")
    model = bigvgan.BigVGAN.from_pretrained(args.pretrained_model, use_cuda_kernel=args.use_cuda_kernel)
    
    # Remove weight norm in the model and set to eval mode
    model.remove_weight_norm()
    model = model.eval().to(device)
    print("Model loaded successfully.")
    
    # Get list of mel spectrogram files
    mel_dir = os.path.join(args.output_dir, 'mels')
    filelist = sorted(os.listdir(mel_dir))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    
    # Process each mel spectrogram
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            if not filename.endswith('.npy'):
                continue
                
            # Load the mel spectrogram
            mel_path = os.path.join(mel_dir, filename)
            print(f"Processing {mel_path}")
            
            # Load the mel spectrogram
            mel = np.load(mel_path)
            mel = torch.FloatTensor(mel).to(device)
            
            # Add batch dimension if needed
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            
            
            # Generate audio
            with torch.inference_mode():
                wav_gen = model(mel)
            
            # Convert to audio
            wav_gen_float = wav_gen.squeeze(0).cpu().numpy().T
            
            # # Save as 16-bit PCM WAV file
            wav_gen_int16 = (wav_gen_float * 32767.0).astype('int16')
            
            # Save the audio file
            output_file = os.path.join(
                args.output_dir, 'audio', os.path.splitext(filename)[0] + ".wav"
            )
            write(output_file, model.h.sampling_rate, wav_gen_int16)
            print(f"Generated: {output_file}")

def main():
    print("Initializing Inference Process..")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="audio", help="generations path containing the mel files and the root path to save generated audio files")
    parser.add_argument("--pretrained_model", default="nvidia/bigvgan_v2_24khz_100band_256x", 
                        help="Pretrained model name or path (default: nvidia/bigvgan_v2_24khz_100band_256x)")
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False, 
                        help="Use CUDA kernel for upsampling")
    
    args = parser.parse_args()
    inference(args)

if __name__ == "__main__":
    main()