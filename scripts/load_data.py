import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from music21 import converter, instrument, note, chord, stream
from tqdm import tqdm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define function to read MIDI files and extract notes
def read_files(file):
    notes = []
    midi = converter.parse(file)
    instrmt = instrument.partitionByInstrument(midi)
    for part in instrmt.parts:
        if 'Piano' in str(part):
            notes_to_parse = part.recurse()
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Load MIDI files
path = r'C:\Users\LENOVO\Desktop\music_gen\data'
all_files = glob.glob(os.path.join(path, '*.mid'), recursive=True)
print(f"Found {len(all_files)} MIDI files.")

notes_array = []

for file in tqdm(all_files, position=0, leave=True):
    try:
        print(f"Reading {file}...")
        data = read_files(file)
        notes_array.append(data)
    except Exception as e:
        print(f"Error reading {file}: {e}")

print("MIDI files read successfully.")
