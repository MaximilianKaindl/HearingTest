# Hearing Test Quiz

A simple audio-based quiz application that tests your ability to identify audio filters applied to pink noise.

## Description

This application generates pink noise and applies various audio filters (Lowpass, Highpass, Notch, and Bandpass). The user listens to both the original and filtered sounds, then tries to identify which filter was applied and at what frequency.

## Features

- Four filter types: Lowpass, Highpass, Notch, and Bandpass
- Pure pink noise generation using the Voss-McCartney algorithm
- Configurable quiz length and audio parameters
- Real-time audio playback
- Scoring system that awards full or partial points based on accuracy

## Requirements

- Python 3.6 or higher
- NumPy
- SciPy
- Sounddevice

## Installation

1. Clone this repository
2. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate HearingTest
```

3. Run the application:

```bash
python test.py
```

## Usage

1. The application will play the original pink noise followed by the filtered version
2. Guess the filter type from: Lowpass, Highpass, Notch, Bandpass
3. If you selected Notch or Bandpass, also guess the center frequency
4. Receive immediate feedback and watch your score grow

## Troubleshooting

If you encounter audio playback issues, ensure your system's audio device is properly configured. On Linux systems, you may need to install PortAudio:

```bash
sudo apt-get install portaudio19-dev  # Debian/Ubuntu
```

Or on macOS:

```bash
brew install portaudio
```

## License

[MIT License](LICENSE)