import glob
import music21

def load_midi_files(dir):
    file_names = glob.glob(dir + "/*.mid")
    midies = []

    for file_name in file_names:
        midi = music21.converter.parse(file_name)
        midies.append(midi)

    return midies, file_names
