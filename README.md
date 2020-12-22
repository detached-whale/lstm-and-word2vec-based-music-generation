# lstm-and-word2vec-based-music-generation
 
This is a simple recurrent neural network to generate music.
<br />
<br />

# Approach

Word2vec is applied to notes(pitch and beat) by considering note as word. Thus, it basically works almost the same way as next word prediction system.
<br />
<br />

# Usage
Put dataset(midi files) to 'data' and run 'main.py'. It will generate midi files into 'output'. It is recommended to use similar musics such as Bach's The Well-Tempered Clavier.
<br />
<br />

# Requirements

* python 3
* music21
* keras
* numpy
* musecore (optional)

