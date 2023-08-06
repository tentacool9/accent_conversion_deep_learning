# Real-Time Accent Conversion
Please refer to original repository as this is a slight modification of the original project. [https://github.com/CorentinJ/Real-Time-Voice-Cloning]

### Papers implemented  
| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |


## Setup

### 1. Install Requirements
1. Both Windows and Linux are supported. A GPU is recommended for training and for inference speed, but is not mandatory.
2. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`, but this is optional.
3. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
4. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
5. Install the remaining requirements with `pip install -r requirements.txt`

### 2. (Optional) Download Pretrained Models
Download models from here: https://drive.google.com/drive/folders/1bnVK9TjbQGOmOYQ1EEoisodUaZPJHHG7?usp=sharing - Save them in the saved_models directory

### 3. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.

### 4. (Optional) Download Datasets


### 5. The model explained
Our final model essentially consisted of 6 networks. Which are actually two neural network pipelines that were duplicated. Our work showed that converting a person’s utterance from one accent to another is a possible feat unlike previous attempts by various research groups. First, we will dive into the architecture of a single neural network pipeline and its components.
Speaker Encoder
In the proposed model, the synthesis network is conditioned on a reference speech signal from the desired target speaker, which is achieved using a speaker encoder. A critical factor for the model's generalization is the utilization of a representation that captures the unique characteristics of different speakers. Additionally, the model should have the ability to identify these characteristics using only a short adaptation signal, independent of its phonetic content and background noise. 
To meet these requirements, a speaker-discriminative model is employed. This model is trained on a text-independent speaker verification task, which provides a highly scalable and accurate neural network for speaker verification. The network is designed to map a sequence of log-mel spectrogram frames, computed from a speech utterance of arbitrary length, to a fixed-dimensional embedding vector. 
The network is trained to optimize a generalized end-to-end speaker verification loss, ensuring that the cosine similarity is high for embeddings of utterances from the same speaker, while maintaining a significant distance in the embedding space for utterances from different speakers. The training dataset consists of speech audio examples segmented into 1.6-second portions and associated speaker identity labels, with no use of transcripts. The input to the network consists of 40-channel log-mel spectrograms which are passed to a network comprising a stack of three LSTM layers of 768 cells each, followed by a projection to 256 dimensions. 

To create the final embedding, the output of the top layer at the final frame is L2-normalized. During inference, an arbitrary length utterance is broken into 800ms windows with an overlap of 50%. The network is run independently on each window and the outputs are then averaged and normalized to create the final utterance embedding. Interestingly, even though the network is not directly optimized to learn a representation that captures speaker characteristics relevant to synthesis, training on a speaker discrimination task leads to an embedding that is suitable for conditioning the synthesis network on speaker identity.

Synthesizer
The next part of the architecture of our model extends the recurrent sequence-to-sequence Tacotron 2 with attention to support multiple speakers. This is accomplished by concatenating an embedding vector for the target speaker with the synthesizer encoder output at each time step. 
The synthesizer is trained on pairs of text transcripts and target audio. At the input, we map the text to a sequence of phonemes, which results in faster convergence and improved pronunciation of rare words and proper nouns. The network is trained in a transfer learning configuration, using a pre-trained speaker encoder to extract a speaker embedding from the target audio. Notably, the speaker reference signal is the same as the target speech during training, and no explicit speaker identifier labels are used. 
Target spectrogram features are computed from 50ms windows with a 12.5ms step. These are then passed through an 80-channel mel-scale filterbank followed by log dynamic range compression. We augment the method by adding an additional L1 loss to the L2 loss on the predicted spectrogram. We found this combined loss to be more robust on noisy training data. Unlike previous works, we do not introduce additional loss terms based on the speaker embedding.
Vocoder
The vocoder utilized to invert synthesized mel spectrograms emitted by the synthesis network into time-domain waveforms is the sample-by-sample autoregressive WaveNet. The architecture is composed of 30 dilated convolution layers and remains consistent with previous descriptions. Notably, the network does not directly condition on the output of the speaker encoder. The mel spectrogram predicted by the synthesizer network encapsulates all the relevant details needed for high-quality synthesis of a variety of voices. 


As stated above, the model consists of two duplicate neural networks. Each model is trained on relevant recordings from different accent oriented datasets. Essentially, we train one model on recordings consisting of only british recordings and we train the other model with recordings of only american accents. This gives each individual network the ability to generate sentences in specific accents, so one can convert anything he says into a different accent. If needed, one can add any speech to text algorithm for a full end-to-end audio recording from one accent to another pipeline.

In our project we chose to leave out the speech to text part as our main contribution is enabling anyone to speak in any accent they want and say whatever they want. 
Training the models consisted in fine tuning the models from the Real Time Voice Cloning implementation of google’s “Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis” paper. We fine tuned the model on data from UK recordings and US recordings separately. This caused each individual TTS pipeline to only know how to generate audio from one accent but not the other while capturing voice features.

