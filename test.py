import numpy as np
import scipy.signal as signal
import sounddevice as sd
import random
import time
import sys
import math

# Basic configuration
FS = 44100  # Sample rate in Hz
DURATION = 3.0  # Duration of each sound clip in seconds
NOISE_AMPLITUDE = 0.4
PAUSE_BETWEEN_SOUNDS = 0.5
PAUSE_BETWEEN_QUESTIONS = 1.5
NUM_QUESTIONS = 10

# Filter definitions
FILTER_TYPES = ["Lowpass", "Highpass", "Notch", "Bandpass"]

# Frequency settings
LOWPASS_FREQ = 5000
HIGHPASS_FREQ = 150

# Notch/Band center frequencies with labels
NOTCH_BAND_CHOICES = {
    500: "Low",
    600: "Low-Mid",
    1500: "Mid",
    5000: "High-Mid",
    8000: "High",
    10000: "Very High",
}
NOTCH_BAND_FREQ_LIST = sorted(list(NOTCH_BAND_CHOICES.keys()))

# Filter design parameters
BUTTER_ORDER = 5
PEAK_EQ_BW_OCT = 1.6
PEAK_EQ_GAIN_DB = 9.0

def generate_pink_noise(duration, fs, n_sources=16):
    """Generates pink noise using Voss-McCartney algorithm."""
    samples = int(duration * fs)
    if samples <= 0:
        return np.array([])

    max_val = 1.0 / n_sources
    sources = np.zeros(n_sources)
    pink = np.zeros(samples)
    acc = 0.0

    for i in range(samples):
        changed = (i + 1) & -(i + 1)
        if changed != 0:
            # Fix the mod operation by adding parentheses
            idx_to_change = ((changed & 0xFFFFFFFF).bit_length() - 1) % n_sources
            acc -= sources[idx_to_change]
            new_val = np.random.uniform(-max_val, max_val)
            sources[idx_to_change] = new_val
            acc += new_val
            
        # Remove white noise contribution
        pink[i] = acc

    # Remove DC offset and normalize
    pink -= np.mean(pink)
    max_abs = np.max(np.abs(pink))
    if max_abs > 1e-9:
        pink /= max_abs
    
    return pink * NOISE_AMPLITUDE

def apply_lowpass(data, cutoff_hz, fs, order=BUTTER_ORDER):
    """Applies a Lowpass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = min(max(cutoff_hz / nyquist, 0.001), 0.999)
    sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    return signal.sosfiltfilt(sos, data)

def apply_highpass(data, cutoff_hz, fs, order=BUTTER_ORDER):
    """Applies a Highpass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = min(max(cutoff_hz / nyquist, 0.001), 0.999)
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    return signal.sosfiltfilt(sos, data)

def design_peak_eq(f0, dBgain, BW_oct, fs):
    """Designs SOS coefficients for a Peaking/Notching EQ filter."""
    # Clamp frequency to valid range
    f0 = max(1.0, min(f0, fs / 2.0 - 1.0))
    BW_oct = max(0.01, BW_oct)

    A = 10**(dBgain / 40.0)
    w0 = 2 * math.pi * f0 / fs
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)

    # Calculate alpha (handle edge cases)
    if abs(sin_w0) < 1e-9:
        Q_approx = 1.0 / (2.0 * math.sinh(math.log(2.0) / 2.0 * BW_oct))
        alpha = sin_w0 / (2.0 * Q_approx)
        if abs(alpha) < 1e-9:
            alpha = 1e-6
    else:
        alpha = sin_w0 * math.sinh(math.log(2.0) / 2.0 * BW_oct * w0 / sin_w0)
    
    alpha = max(1e-6, min(alpha, 0.99))

    # Calculate coefficients
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    # Check for stability
    if abs(a0) < 1e-9:
        return np.array([[1., 0., 0., 1., 0., 0.]])

    # Normalize and convert to SOS
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    
    try:
        sos = signal.tf2sos(b, a, analog=False)
        return sos
    except ValueError:
        return np.array([[1., 0., 0., 1., 0., 0.]])

def apply_biquad_filter(data, sos):
    """Applies a filter defined by SOS coefficients."""
    if sos is None or sos.shape[0] == 0:
        return data
    return signal.sosfiltfilt(sos, data.astype(np.float64))

def run_quiz():
    """Runs the hearing test quiz."""
    print("-" * 30)
    print(" Hearing Test Quiz")
    print("-" * 30)
    print(f"Each question will play original pink noise, then filtered pink noise.")
    print(f"1. Guess the filter type: {', '.join(FILTER_TYPES)}")
    print(f"2. If Notch or Bandpass, guess the center frequency.")
    print("-" * 30)
    print("Scoring:")
    print(" - Lowpass/Highpass: 1 point for correct type.")
    print(" - Notch/Bandpass: 0.5 points for correct type, wrong frequency.")
    print(" - Notch/Bandpass: 1 point for correct type AND correct frequency.")
    print("-" * 30)

    # Get user preferences
    show_filter_type_ans = get_yes_no_input("Show correct FILTER TYPE after each guess? (y/n): ")
    show_details_ans = get_yes_no_input("Show filter DETAILS (Freq, BW, Gain) after each guess? (y/n): ")

    print("\nStarting the quiz...")
    time.sleep(1)

    score = 0.0
    
    for i in range(NUM_QUESTIONS):
        print(f"\n--- Question {i+1}/{NUM_QUESTIONS} ---")

        # Choose filter type and settings
        filter_type = random.choice(FILTER_TYPES)
        correct_freq, correct_label, details_str = get_filter_settings(filter_type)

        # Generate and filter noise
        original_noise = generate_pink_noise(DURATION, FS)
        filtered_noise = apply_filter(original_noise, filter_type, correct_freq)

        # Ensure filtered noise is valid
        if filtered_noise is None or np.max(np.abs(filtered_noise)) < 1e-6:
            filtered_noise = original_noise
        
        # Play audio
        try:
            play_audio(original_noise, filtered_noise)
        except Exception as e:
            handle_audio_error(e)
            sys.exit(1)

        # Get user guess
        user_guess_type = get_filter_type_guess()
        user_guess_freq = None
        
        if user_guess_type in ["Notch", "Bandpass"]:
            user_guess_freq = get_frequency_guess()

        # Score and feedback
        current_question_score, correct_type, correct_freq_guess = score_answer(
            filter_type, correct_freq, user_guess_type, user_guess_freq
        )
        
        score += current_question_score
        
        # Provide feedback
        give_feedback(
            correct_type, correct_freq_guess, filter_type, 
            correct_freq, correct_label, details_str,
            show_filter_type_ans, show_details_ans, 
            user_guess_type, score, i
        )
        
        time.sleep(PAUSE_BETWEEN_QUESTIONS)

    # Quiz completion
    print("\n" + "=" * 30)
    print("      Quiz Complete!")
    print(f"      Final Score: {score:.1f}/{NUM_QUESTIONS}")
    print("=" * 30)

def get_yes_no_input(prompt):
    """Gets a yes/no input from the user."""
    while True:
        response = input(prompt).lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")

def get_filter_settings(filter_type):
    """Gets the correct settings for a given filter type."""
    if filter_type == "Lowpass":
        return LOWPASS_FREQ, "Lowpass", f"Cutoff: {LOWPASS_FREQ} Hz (Butterworth Order {BUTTER_ORDER})"
    elif filter_type == "Highpass":
        return HIGHPASS_FREQ, "Highpass", f"Cutoff: {HIGHPASS_FREQ} Hz (Butterworth Order {BUTTER_ORDER})"
    else:  # Notch or Bandpass
        freq = random.choice(NOTCH_BAND_FREQ_LIST)
        label = NOTCH_BAND_CHOICES[freq]
        gain = -PEAK_EQ_GAIN_DB if filter_type == "Notch" else PEAK_EQ_GAIN_DB
        return freq, label, f"Center: {freq} Hz ({label}), BW: {PEAK_EQ_BW_OCT:.1f} Oct, Gain: {gain:+.1f} dB"

def apply_filter(audio, filter_type, frequency):
    """Applies the specified filter to the audio."""
    if filter_type == "Lowpass":
        return apply_lowpass(audio, frequency, FS)
    elif filter_type == "Highpass":
        return apply_highpass(audio, frequency, FS)
    elif filter_type == "Notch":
        sos = design_peak_eq(frequency, -PEAK_EQ_GAIN_DB, PEAK_EQ_BW_OCT, FS)
        return apply_biquad_filter(audio, sos)
    elif filter_type == "Bandpass":
        sos = design_peak_eq(frequency, PEAK_EQ_GAIN_DB, PEAK_EQ_BW_OCT, FS)
        return apply_biquad_filter(audio, sos)
    return audio

def play_audio(original, filtered):
    """Plays the original and filtered audio samples."""
    print("Playing ORIGINAL noise...")
    sys.stdout.flush()
    sd.play(original, FS)
    sd.wait()
    time.sleep(PAUSE_BETWEEN_SOUNDS)

    print("Playing FILTERED noise...")
    sys.stdout.flush()
    sd.play(filtered, FS)
    sd.wait()

def handle_audio_error(error):
    """Handles errors during audio playback."""
    print(f"\nError during audio playback: {error}")
    print("Please ensure you have a working audio output device and 'sounddevice' is installed correctly.")
    print("You might need to install PortAudio ('sudo apt-get install portaudio19-dev' on Debian/Ubuntu, 'brew install portaudio' on macOS).")

def get_filter_type_guess():
    """Gets the user's guess for the filter type."""
    while True:
        raw_guess = input(f"Guess the filter type ({', '.join(FILTER_TYPES)}): ").strip().lower()
        
        if raw_guess in ['lp', 'low', 'lowpass']: 
            return 'Lowpass'
        elif raw_guess in ['hp', 'high', 'highpass']: 
            return 'Highpass'
        elif raw_guess in ['n', 'notch']: 
            return 'Notch'
        elif raw_guess in ['bp', 'band', 'bandpass']: 
            return 'Bandpass'
        
        user_guess = raw_guess.capitalize()
        if user_guess in FILTER_TYPES:
            return user_guess
        
        print(f"Invalid guess. Please enter one of: {', '.join(FILTER_TYPES)} (or abbreviations like lp, hp, n, bp)")

def get_frequency_guess():
    """Gets the user's guess for the center frequency."""
    choice_strs = [f"{f} Hz ({NOTCH_BAND_CHOICES[f]})" for f in NOTCH_BAND_FREQ_LIST]
    print(f"Available frequencies: {', '.join(choice_strs)}")

    while True:
        freq_guess_str = input(f"Guess the center frequency (enter just the number): ").strip()
        try:
            guess = int(freq_guess_str)
            if guess in NOTCH_BAND_FREQ_LIST:
                return guess
            print(f"Invalid frequency. Please enter one of: {NOTCH_BAND_FREQ_LIST}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def score_answer(filter_type, correct_freq, user_guess_type, user_guess_freq):
    """Scores the user's answer and returns feedback."""
    correct_type = (user_guess_type == filter_type)
    correct_freq_guess = False
    score = 0.0
    
    if filter_type in ["Lowpass", "Highpass"]:
        if correct_type:
            score = 1.0
    elif filter_type in ["Notch", "Bandpass"]:
        if correct_type:
            if user_guess_freq == correct_freq:
                score = 1.0
                correct_freq_guess = True
            else:
                score = 0.5
    
    return score, correct_type, correct_freq_guess

def give_feedback(correct_type, correct_freq_guess, filter_type, 
                 correct_freq, correct_label, details_str,
                 show_filter_type_ans, show_details_ans, 
                 user_guess_type, score, question_num):
    """Gives feedback on the user's answer."""
    if not correct_type:
        print("Incorrect.")
    elif filter_type in ["Notch", "Bandpass"] and not correct_freq_guess:
        print(f"Partially Correct. (Correct type '{filter_type}', but wrong frequency)")
    elif filter_type in ["Notch", "Bandpass"] and correct_freq_guess:
        print("Correct! (Type and Frequency)")
    else:
        print("Correct!")

    feedback = []
    if show_filter_type_ans:
        if not correct_type:
            feedback.append(f"The correct filter type was: {filter_type}")
        if filter_type in ["Notch", "Bandpass"] and not correct_freq_guess and user_guess_type == filter_type:
            feedback.append(f"The correct frequency was: {correct_freq} Hz ({correct_label})")

    if show_details_ans:
        feedback.append(f"Filter Details: {details_str}")

    if feedback:
        print(" | ".join(feedback))

    print(f"Current Score: {score:.1f}/{question_num+1}")

if __name__ == "__main__":
    try:
        # Quick audio device check
        try:
            print("Testing audio device...")
            sd.play(np.zeros(10), FS, blocking=True)
        except Exception as e:
            print(f"\nWarning: Audio device issue: {e}")
            print("Please check your audio settings and connections.")
            time.sleep(2)
            
        run_quiz()
    except KeyboardInterrupt:
        print("\nQuiz interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sd.stop()