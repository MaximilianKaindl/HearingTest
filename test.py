import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal
import time
import random

# Sample rate
fs = 44100  # 44.1 kHz

# Test parameters
duration = 3  # seconds per test
volume = 0.3  # adjust as needed
show_filter_type = False  # Set to True to show the filter type to the testee

# Frequency labels
frequency_labels = {
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

# Reverse mapping for user input
label_to_frequency = {
    "lowpass": 125,
    "highpass": 250,
    "low": 500,
    "lowmid": 1000,
    "mid": 2000,
    "highmid": 4000,
    "high": 8000,  # Both 8000 and 10000 map to "High"
    "very high": 12000  # Both 12000 and 16000 map to "Very High"
}

# Generate pink noise with highpass and lowpass filters
def generate_pink_noise(duration, fs):
    """Generate pink noise with highpass at 150Hz and lowpass at 5000Hz."""
    samples = int(duration * fs)
    white_noise = np.random.normal(0, 1, samples)
    
    # Create pink noise by applying 1/f filter
    b, a = signal.butter(1, 0.5, 'lowpass')
    pink_noise = signal.lfilter(b, a, white_noise)
    
    # Apply highpass filter at 150Hz
    nyquist = fs / 2
    high_cutoff = 150 / nyquist
    b_high, a_high = signal.butter(4, high_cutoff, 'highpass')
    pink_noise = signal.filtfilt(b_high, a_high, pink_noise)
    
    # Apply lowpass filter at 5000Hz
    low_cutoff = 5000 / nyquist
    b_low, a_low = signal.butter(4, low_cutoff, 'lowpass')
    pink_noise = signal.filtfilt(b_low, a_low, pink_noise)
    
    # Normalize
    pink_noise = pink_noise / np.max(np.abs(pink_noise)) * volume
    
    return pink_noise

# Apply notch or bandpass filter to pink noise
def apply_filter(noise, filter_type, center_freq, fs, width=0.5, boost_db=9):
    """Apply notch or bandpass filter to noise."""
    # Normalisierte Frequenzen berechnen (Nyquist = 1.0)
    nyquist = fs / 2
    center_norm = center_freq / nyquist
    
    # Bandbreite als Anteil der Mittenfrequenz
    bandwidth = width * center_freq
    
    # Untere und obere Grenzfrequenz berechnen
    low_freq = max(0.001, center_freq - bandwidth)
    high_freq = min(nyquist - 1, center_freq + bandwidth)
    
    # Normalisierte Grenzfrequenzen
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Sicherstellen, dass low_norm < high_norm
    if low_norm >= high_norm:
        low_norm = max(0.001, high_norm * 0.5)
    
    if filter_type == 'notch':
        # Kerbfilter (entfernt Frequenzband)
        b, a = signal.iirnotch(center_freq, 30, fs)
        filtered_noise = signal.filtfilt(b, a, noise)
        
        # Calculate frequency response of the filter
        w, h = signal.freqz(b, a, fs=fs)
        
        # Find the frequencies that were most affected by the notch
        notch_band = (w >= low_freq) & (w <= high_freq)
        
        # Create a boost filter around the notch
        # Convert dB boost to amplitude factor
        boost_factor = 10**(boost_db/20)
        
        # Create a frequency-dependent gain array
        gain = np.ones(len(noise))
        
        # Apply FFT
        noise_fft = np.fft.rfft(noise)
        freq = np.fft.rfftfreq(len(noise), 1/fs)
        
        # Apply boost around the notch frequency
        boost_band = ((freq >= center_freq - bandwidth) & 
                      (freq <= center_freq + bandwidth))
        noise_fft[boost_band] *= boost_factor
        
        # Apply IFFT to get back to time domain
        boosted_noise = np.fft.irfft(noise_fft, len(noise))
        
        # Mix the notch-filtered and boosted signal
        filtered_noise = boosted_noise
    else:  # bandpass
        # Bandpassfilter (behält nur Frequenzband)
        b, a = signal.butter(4, [low_norm, high_norm], 'bandpass')
        filtered_noise = signal.filtfilt(b, a, noise)
        
        # Apply boost to the bandpass
        boost_factor = 10**(boost_db/20)
        filtered_noise *= boost_factor
    
    # Normalize to prevent clipping
    filtered_noise = filtered_noise / max(1.0, np.max(np.abs(filtered_noise))) * volume
    
    return filtered_noise

# Plot the frequency spectrum
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

# Run the hearing test
def run_hearing_test(show_filter_type=show_filter_type):
    print("\n===== HEARING TEST =====")
    print("You will hear 10 sound samples with filtered pink noise.")
    if show_filter_type:
        print("For each sample, you'll be told which type of filter is applied.")
    print("Your task is to identify which frequency range is affected.")
    print("\nAfter each sound plays, enter the frequency range you think was affected.")
    print("Available options:")
    print("  - Lowpass (125 Hz)")
    print("  - Highpass (250 Hz)")
    print("  - Low (500 Hz)")
    print("  - Lowmid (1000 Hz)")
    print("  - Mid (2000 Hz)")
    print("  - Highmid (4000 Hz)")
    print("  - High (8000-10000 Hz)")
    print("  - Very High (12000-16000 Hz)")
    print("\nPress Enter to start the test...\n")
    input()
    
    # Prepare test questions
    questions = []
    correct_answers = []
    
    # Use all frequencies
    frequencies = list(frequency_labels.keys())
    
    for i in range(10):
        # Randomly select frequency, but use only notch filter
        freq = random.choice(frequencies)
        filter_type = 'notch'  # Only use notch filter
        
        # Generate base pink noise
        noise = generate_pink_noise(duration, fs)
        
        # Apply selected filter
        filtered_noise = apply_filter(noise, filter_type, freq, fs)
        
        # Store question info
        questions.append({
            'audio': filtered_noise,
            'filter_type': filter_type,
            'frequency': freq,
            'freq_label': frequency_labels[freq]  # Store the frequency label
        })
        
        correct_answers.append(frequency_labels[freq].lower())
    
    # Run the test
    user_answers = []
    score = 0
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}/10:")
        
        # First play the original pink noise
        print("Playing original pink noise...")
        original_noise = generate_pink_noise(duration, fs)
        sd.play(original_noise, fs)
        sd.wait()
        
        # Short pause
        time.sleep(1)
        
        # Show filter type information if enabled
        if show_filter_type:
            print(f"Now playing sound with {question['filter_type']} filter applied.")
        else:
            print("Now playing modified sound...")
            
        print("Which frequency range is affected?")
        
        # Play the filtered audio
        sd.play(question['audio'], fs)
        sd.wait()
        
        # Get user answer with improved input handling
        valid_inputs = ["lowpass", "highpass", "low", "lowmid", "mid", "highmid", "high", "very high"]
        print("Enter one of: Lowpass, Highpass, Low, Lowmid, Mid, Highmid, High, Very High")
        
        while True:
            answer = input("> ").strip().lower()
            # Remove spaces and normalize "very high" to "veryhigh" for easier comparison
            answer_normalized = answer.replace(" ", "")
            
            # Check if input matches any valid option, allowing for variation
            if answer in valid_inputs:
                # Direct match
                break
            elif "very" in answer and "high" in answer:
                answer = "very high"
                break
            elif answer_normalized in ["highpass", "hp"]:
                answer = "highpass"
                break
            elif answer_normalized in ["lowpass", "lp"]:
                answer = "lowpass"
                break
            elif answer_normalized in ["highmid", "highm", "hmid"]:
                answer = "highmid"
                break
            elif answer_normalized in ["lowmid", "lowm", "lmid"]:
                answer = "lowmid"
                break
            elif answer_normalized in ["veryhigh", "vhigh", "vh"]:
                answer = "very high"
                break
            else:
                print("Invalid input. Please enter one of the frequency ranges:")
                print("Lowpass, Highpass, Low, Lowmid, Mid, Highmid, High, Very High")
        
        user_answers.append(answer)
        
        # Check answer against correct frequency label
        correct = answer.lower() == question['freq_label'].lower()
        if correct:
            score += 1
            print("✓ Correct!")
        else:
            print(f"✗ Wrong. The correct frequency range was {question['freq_label']}.")
        
        # Optional: Show spectrum
        # plot_spectrum(question['audio'], fs, f"Spectrum - {question['filter_type']} at {question['frequency']} Hz")
    
    # Show final results
    print("\n===== TEST RESULTS =====")
    print(f"Your score: {score}/10")
    
    print("\nQuestion details:")
    for i in range(10):
        print(f"Q{i+1}: {questions[i]['filter_type']} at {questions[i]['frequency']} Hz ({questions[i]['freq_label']}) - " + 
              ("Correct" if user_answers[i].lower() == correct_answers[i] else f"Wrong (you answered {user_answers[i]})"))
    
    # Hearing capabilities assessment
    if score >= 9:
        print("\nExcellent hearing! You can detect subtle frequency changes very well.")
    elif score >= 7:
        print("\nGood hearing. You can detect most frequency changes.")
    elif score >= 5:
        print("\nAverage hearing. You might want to pay attention to certain frequency ranges.")
    else:
        print("\nYou might have some difficulty distinguishing frequencies.")
        print("Consider taking a professional hearing test.")

if __name__ == "__main__":
    try:
        # Uncomment to test the audio generation
        # noise = generate_pink_noise(3, fs)
        # filtered = apply_filter(noise, 'notch', 1000, fs)
        # plot_spectrum(filtered, fs, "Test Spectrum")
        # sd.play(filtered, fs)
        # sd.wait()
        
        # Run the hearing test
        run_hearing_test(show_filter_type)
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"\nError: {e}")