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
import re
from pydub import AudioSegment
import base64
import hashlib
## Md5 the request id so that it is a consistent length and is shorter
#
#print()
#####

# Set up server
from flask import Flask, jsonify, request, abort,send_file, send_from_directory
app = Flask(__name__)

@app.route('/getResult')
def getResult():
    index = int(request.args.get('index'))
    req_id = request.args.get('req_id')
    return send_from_directory("/output/%s/" % req_id, "output_%03d.wav" % index)

@app.route('/getCombinedResult')
def getCombinedResult():
    req_id = request.args.get('req_id')
    
    output_path = "/output/%s/" % req_id
    merged_output = None

    if os.path.exists(output_path):
        files = [f for f in os.listdir(output_path) if f.startswith("output_") and f.endswith(".wav")]
        files.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f)))))
        for file in files:
            print(file)
            file_path = os.path.join(output_path, file)
            sound_segment = AudioSegment.from_wav(file_path)
            if merged_output is None:
                merged_output = sound_segment
            else:
                merged_output = merged_output + sound_segment
    merged_output_path = os.path.join(output_path, "merged.wav")
    merged_output.export(merged_output_path, format="wav")
    return send_from_directory(output_path, "merged.wav")

@app.route("/clone_voices", methods=["POST"])
def run_voice_cloning():
    ## Model locations
    enc_model_fpath = Path("encoder/saved_models/pretrained.pt")
    syn_model_dir = Path("synthesizer/saved_models/logs-pretrained/")
    voc_model_fpath = Path("vocoder/saved_models/pretrained/pretrained.pt")
    ref_voice_path = request.json["voiceFile"] # filename like ojo3.wav
    messages = request.json["messages"] # array of strings
    low_mem = request.json["low_mem"] if "low_mem" in request.json else False # whether to use LowMem Mode

    # Base64 encode the parameters so that we can reference this job in later api calls
    dataToEncodeAsID = ','.join(messages) + ref_voice_path
    encodedBytes = base64.b64encode(dataToEncodeAsID.encode("utf-8"))
    req_id = str(encodedBytes, "utf-8")
    # Md5 Hash it so that it is a consistent length
    req_id = hashlib.md5(req_id.encode('utf-8')).hexdigest()

    # Clear destination folder of generated sound files
    output_path = "/output/%s/" % req_id
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
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
    synthesizer = Synthesizer(syn_model_dir.joinpath("taco_pretrained"), low_mem=low_mem)
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

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")

    return req_id
        