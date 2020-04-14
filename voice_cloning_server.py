from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import torch
import sys
import os
import shutil

# Set up server
from flask import Flask, jsonify, request, abort,send_file, send_from_directory
app = Flask(__name__)

@app.route('/lastGeneratedWav')
def lastGeneratedWav():
    return send_from_directory('/output/','output_000.wav')

@app.route("/clone_voices", methods=["POST"])
def run_voice_cloning():
    ## Model locations
    enc_model_fpath = Path("encoder/saved_models/pretrained.pt")
    syn_model_dir = Path("synthesizer/saved_models/logs-pretrained/")
    voc_model_fpath = Path("vocoder/saved_models/pretrained/pretrained.pt")
    ref_voice_path = request.json["voiceFile"] # filename like ojo3.wav
    messages = request.json["messages"] # array of strings

    # Clear destination folder of generated sound files
    output_path = "/output/"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        return abort(500)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
    
    
    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(enc_model_fpath)
    synthesizer = Synthesizer(syn_model_dir.joinpath("taco_pretrained"), low_mem=False)
    vocoder.load_model(voc_model_fpath)
        
    in_fpath = Path(ref_voice_path)
    
    print("Computing the embedding")
    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is 
    # important: there is preprocessing that must be applied.
    
    # The following two methods are equivalent:
    # - Directly load from the filepath:
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(in_fpath)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("Loaded file succesfully")
    
    # Then we derive the embedding. There are many functions and parameters that the 
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")

    print("Generation loop")
    num_generated = 0
    fpath = None
    for text in messages:
        try:
            
            
            ## Generating the spectrogram
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")
            
            
            ## Generating the waveform
            print("Synthesizing the waveform:")
            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)
            
            
            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            
            # Save it on the disk
            fpath = output_path + ("output_%03d.wav" % num_generated)
            print(generated_wav.dtype)
            librosa.output.write_wav(fpath, generated_wav.astype(np.float32), 
                                     synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % fpath)

            
            # TODO: Convert to OGG
            
        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")

    return str(num_generated)
        