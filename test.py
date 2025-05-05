import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal
import time
import random

# Constants
SAMPLE_RATE = 44100  # 44.1 kHz
TEST_DURATION = 1  # seconds per test
VOLUME = 0.3  # audio volume
SHOW_FILTER_TYPE = False  # whether to show filter type to user

# Frequency bands with their labels
FREQUENCY_BANDS = {
    125: "Lowpass",
    250: "Highpass",
    500: "Low",
    1000: "Lowmid",
    2000: "Mid",
    4000: "Highmid", 
    8000: "High",
    10000: "High",
    12000: "Very High",
    16000: "Very High"
}

# Mapping for user input to frequency
LABEL_TO_FREQUENCY = {
    "lowpass": 125,
    "highpass": 250,
    "low": 500,
    "lowmid": 1000,
    "mid": 2000,
    "highmid": 4000,
    "high": 8000,  # Both 8000 and 10000 map to "High"
    "very high": 12000  # Both 12000 and 16000 map to "Very High"
}

# Valid user inputs with aliases
VALID_INPUTS = {
    "lowpass": ["lowpass", "lp"],
    "highpass": ["highpass", "hp"],
    "low": ["low"],
    "lowmid": ["lowmid", "lowm", "lmid"],
    "mid": ["mid"],
    "highmid": ["highmid", "highm", "hmid"],
    "high": ["high"],
    "very high": ["very high", "veryhigh", "vhigh", "vh"]
}

def generate_pink_noise(duration, fs):
    """Generate pink noise with highpass at 150Hz and lowpass at 5000Hz."""
    samples = int(duration * fs)
    white_noise = np.random.normal(0, 1, samples)
    
    # Create pink noise using 1/f filter
    b, a = signal.butter(1, 0.5, 'lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)
    
    # Apply highpass and lowpass filters
    nyquist = fs / 2
    pink_noise = apply_bandpass(pink_noise, 150, 5000, fs)
    
    # Normalize
    return normalize_audio(pink_noise, VOLUME)

def apply_bandpass(audio, low_freq, high_freq, fs):
    """Apply bandpass filter between low_freq and high_freq."""
    nyquist = fs / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    b, a = signal.butter(4, [low_norm, high_norm], 'bandpass')
    return signal.filtfilt(b, a, audio)

def normalize_audio(audio, target_volume):
    """Normalize audio to prevent clipping."""
    return audio / np.max(np.abs(audio)) * target_volume

def apply_filter(noise, center_freq, fs, width=0.5, boost_db=9):
    """Apply notch filter with boost around the center frequency."""
    nyquist = fs / 2
    
    # Calculate bandwidth
    bandwidth = width * center_freq
    
    # Create notch filter
    b, a = signal.iirnotch(center_freq, 30, fs)
    
    # Apply FFT
    noise_fft = np.fft.rfft(noise)
    freq = np.fft.rfftfreq(len(noise), 1/fs)
    
    # Apply boost around the notch frequency
    boost_factor = 10**(boost_db/20)
    boost_band = ((freq >= center_freq - bandwidth) & 
                  (freq <= center_freq + bandwidth))
    noise_fft[boost_band] *= boost_factor
    
    # Apply IFFT to get back to time domain
    boosted_noise = np.fft.irfft(noise_fft, len(noise))
    
    # Normalize to prevent clipping
    return normalize_audio(boosted_noise, VOLUME)

def plot_spectrum(audio, fs, title):
    """Plot the frequency spectrum of an audio signal."""
    plt.figure(figsize=(10, 6))
    
    # Calculate FFT
    n = len(audio)
    yf = np.abs(np.fft.rfft(audio))
    xf = np.fft.rfftfreq(n, 1 / fs)
    
    # Plot on log scale for frequency
    plt.semilogx(xf, 20 * np.log10(yf))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title(title)
    plt.xlim(20, 20000)  # Human hearing range
    
    # Add vertical lines for common frequencies
    for freq in [125, 250, 500, 1000, 2000, 4000, 8000, 16000]:
        plt.axvline(x=freq, color='r', linestyle='--', alpha=0.3)
        plt.text(freq, plt.ylim()[1]*0.9, f"{freq}Hz", rotation=90, alpha=0.7)
    
    plt.show()

def get_user_answer():
    """Get and validate user input for frequency band."""
    print("Enter one of: Lowpass, Highpass, Low, Lowmid, Mid, Highmid, High, Very High")
    
    while True:
        user_input = input("> ").strip().lower()
        
        # Check if input matches any valid option
        for answer, aliases in VALID_INPUTS.items():
            if user_input in aliases or user_input.replace(" ", "") in [a.replace(" ", "") for a in aliases]:
                return answer
        
        print("Invalid input. Please enter one of the frequency ranges:")
        print("Lowpass, Highpass, Low, Lowmid, Mid, Highmid, High, Very High")

def run_hearing_test(show_filter_type=SHOW_FILTER_TYPE):
    """Run the complete hearing test with 10 questions."""
    print("\n===== HEARING TEST =====")
    print("You will hear 10 sound samples with filtered pink noise.")
    if show_filter_type:
        print("For each sample, you'll be told which type of filter is applied.")
    print("Your task is to identify which frequency range is affected.")
    print("\nAfter each sound plays, enter the frequency range you think was affected.")
    print("Available options:")
    for freq, label in sorted(FREQUENCY_BANDS.items()):
        if freq in [125, 250, 500, 1000, 2000, 4000, 8000, 12000]:
            print(f"  - {label} ({freq} Hz)")
    
    print("\nPress Enter to start the test...\n")
    input()
    
    # Prepare test questions with selected frequencies
    test_frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 10000, 12000, 16000]
    selected_frequencies = random.sample(test_frequencies, 10)
    
    user_answers = []
    score = 0
    
    # Run through each question
    for i, freq in enumerate(selected_frequencies):
        print(f"\nQuestion {i+1}/10:")
        
        # Play original pink noise as reference
        print("Playing original pink noise...")
        original_noise = generate_pink_noise(TEST_DURATION, SAMPLE_RATE)
        sd.play(original_noise, SAMPLE_RATE)
        sd.wait()
        
        # Short pause
        time.sleep(1)
        
        # Generate filtered noise
        filtered_noise = apply_filter(original_noise, freq, SAMPLE_RATE)
        
        # Show filter type if enabled
        if show_filter_type:
            print(f"Now playing sound with notch filter applied.")
        else:
            print("Now playing modified sound...")
        
        print("Which frequency range is affected?")
        
        # Play the filtered audio
        sd.play(filtered_noise, SAMPLE_RATE)
        sd.wait()
        
        # Get user answer
        answer = get_user_answer()
        user_answers.append(answer)
        
        # Check answer
        correct_label = FREQUENCY_BANDS[freq].lower()
        correct = answer.lower() == correct_label
        
        if correct:
            score += 1
            print("✓ Correct!")
        else:
            print(f"✗ Wrong. The correct frequency range was {FREQUENCY_BANDS[freq]}.")
    
    # Show final results
    print("\n===== TEST RESULTS =====")
    print(f"Your score: {score}/10")
    
    print("\nQuestion details:")
    for i, freq in enumerate(selected_frequencies):
        correct = user_answers[i].lower() == FREQUENCY_BANDS[freq].lower()
        result = "Correct" if correct else f"Wrong (you answered {user_answers[i]})"
        print(f"Q{i+1}: Notch at {freq} Hz ({FREQUENCY_BANDS[freq]}) - {result}")
    
    # Hearing assessment
    if score >= 9:
        print("\nExcellent hearing! You can detect subtle frequency changes very well.")
    elif score >= 7:
        print("\nGood hearing. You can detect most frequency changes.")
    elif score >= 5:
        print("\nAverage hearing. You might want to pay attention to certain frequency ranges.")
    else:
        print("\nYou might have some difficulty distinguishing frequencies.")

if __name__ == "__main__":
    try:
        run_hearing_test(SHOW_FILTER_TYPE)
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"\nError: {e}")