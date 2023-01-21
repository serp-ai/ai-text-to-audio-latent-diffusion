import torch
import torchaudio
from torchaudio import transforms as T
import random
from glob import glob
import os
import re
from audio_diffusion.utils import Stereo, PadCrop, RandomPhaseInvert
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
import torchaudio
from torchaudio import transforms as T
import random
from glob import glob
import os
from audio_diffusion.utils import Stereo, PadCrop, RandomPhaseInvert
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from encodec.utils import convert_audio

class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, paths, global_args, tokenizer=None):
    super().__init__()
    self.filenames = []

    print(f"Random crop: {global_args.random_crop}")
    self.augs = torch.nn.Sequential(
      PadCrop(global_args.sample_size, randomize=global_args.random_crop),
      # RandomPhaseInvert(),
    )

    if tokenizer is not None:
      self.tokenizer = tokenizer
    else:
      self.tokenizer = None

    for path in paths:
      for ext in ['wav','flac','ogg','aiff','aif','mp3']:
        self.filenames += glob(f'{path}/**/*.{ext}', recursive=True)

    self.sr = global_args.sample_rate
    if hasattr(global_args,'load_frac'):
      self.load_frac = global_args.load_frac
    else:
      self.load_frac = 1.0
    self.num_gpus = global_args.num_gpus

    # if hasattr(global_args,'channels'):
    #   self.channels = global_args.channels
    # else:
    #   self.channels = 2
    self.channels = 1

    self.use_text_dropout = global_args.use_text_dropout
    self.text_dropout_prob = global_args.text_dropout_prob

    self.shuffle_prompts = global_args.shuffle_prompts
    self.shuffle_prompts_sep = global_args.shuffle_prompts_sep.strip("\'").strip('\"')
    self.shuffle_prompts_prob = global_args.shuffle_prompts_prob

    self.cache_training_data = global_args.cache_training_data

    if self.cache_training_data: self.preload_files()


  def load_file(self, filename):
    wav, sr = torchaudio.load(filename)
    wav = torch.mean(wav, dim=0, keepdim=True) # convert to 1 channel
    wav = wav.unsqueeze(0)
    audio = convert_audio(wav, sr, self.sr, self.channels)
    return audio

  def load_file_ind(self, file_list,i): # used when caching training data
    return self.load_file(file_list[i]).cpu()

  def get_data_range(self): # for parallel runs, only grab part of the data
    start, stop = 0, len(self.filenames)
    try: 
      local_rank = int(os.environ["LOCAL_RANK"])
      world_size = int(os.environ["WORLD_SIZE"])
      interval = stop//world_size 
      start, stop = local_rank*interval, (local_rank+1)*interval
      print("local_rank, world_size, start, stop =",local_rank, world_size, start, stop)
      return start, stop
    except KeyError as e: # we're on GPU 0 and the others haven't been initialized yet
      start, stop = 0, len(self.filenames)//self.num_gpus
      return start, stop

  def preload_files(self):
      n = int(len(self.filenames)*self.load_frac)
      print(f"Caching {n} input audio files:")
      wrapper = partial(self.load_file_ind, self.filenames)
      start, stop = self.get_data_range()
      with Pool(processes=cpu_count()) as p:   # //8 to avoid FS bottleneck and/or too many processes (b/c * num_gpus)
        self.audio_files = list(tqdm.tqdm(p.imap(wrapper, range(start,stop)), total=stop-start))

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    try:
      if self.cache_training_data:
        audio = self.audio_files[idx] # .copy()
      else:
        audio = self.load_file(audio_filename)

      if len(audio.shape) > 2:
        audio = audio.squeeze(0)
      #Run augmentations on this sample (including random crop)
      if self.augs is not None:
        audio = self.augs(audio)

      if self.use_text_dropout:
        if random.random() < self.text_dropout_prob:
          return (audio, '')
      audio_filename = os.path.splitext(audio_filename)[0].lower().split('/')[-1].split('\\')[-1]
      # remove _chunk{num} from filename
      audio_filename = re.sub(r'_chunk\d+', '', audio_filename)
      if self.shuffle_prompts and random.random() < self.shuffle_prompts_prob:
        # split the filename by seperator and shuffle the order
        audio_filename = audio_filename.split(self.shuffle_prompts_sep)
        random.shuffle(audio_filename)
        audio_filename = self.shuffle_prompts_sep.join(audio_filename)
      return (audio, audio_filename)
    except Exception as e:
      print(f'Couldn\'t load file {audio_filename}: {e}')
      return self[random.randrange(len(self))]
