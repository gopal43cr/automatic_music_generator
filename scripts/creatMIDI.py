from generate import *
# Create MIDI function
import os

# Define the create_midi function
def create_midi(generated_notes, output_file):
    output_notes = []

    for offset, note_str in enumerate(generated_notes):
        if '.' in note_str or note_str.isdigit():
            chord_notes = [note.Note(int(n)) for n in note_str.split('.')]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(note_str)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

# Define the output path
path = r'C:\Users\LENOVO\Desktop\music_gen\output'
os.makedirs(path, exist_ok=True)

# Function to generate multiple MIDI files
def generate_multiple_midi_files(generated_notes, output_folder=path, num_files=5):
    for i in range(1, num_files + 1):
        output_file = os.path.join(output_folder, f'generated_music_{i}.mid')
        create_midi(generated_notes, output_file)
        print(f"MIDI file {i} generated and saved successfully at {output_file}.")

# Generate 5 MIDI files in the specified path directory
generate_multiple_midi_files(generated_notes, output_folder=path, num_files=5)
