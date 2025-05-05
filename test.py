import numpy as np
import scipy.signal as signal
import sounddevice as sd
import random
import time
import sys # To flush output buffer
import math # For log, sin, cos, sinh in EQ design

# --- Configuration ---
FS = 44100  # Sample rate in Hz
DURATION = 3.0  # Duration of each sound clip in seconds
NOISE_AMPLITUDE = 0.4 # Amplitude to prevent clipping
PAUSE_BETWEEN_SOUNDS = 0.5 # Pause between original and filtered sound
PAUSE_BETWEEN_QUESTIONS = 1.5 # Pause after feedback before next question
NUM_QUESTIONS = 10

# --- Filter Definitions ---
# Store frequencies primarily for internal use during filter *creation*
INTERNAL_FILTER_FREQS = {
    # Pass Filters (using specific frequencies from the prompt)
    5000: "Lowpass",  # Cutoff for Lowpass
    150: "Highpass", # Cutoff for Highpass

    # Frequencies for Notches & Bands (using specific frequencies + labels)
    # These are the *potential* center frequencies the program can *choose*
    500: "Low",
    600: "Low-Mid",
    1500: "Mid",
    5000: "High-Mid", # 5000Hz can be LP cutoff OR Notch/Band center
    8000: "High",
    10000: "Very High",
}

# Define the filter *types* the user can guess
FILTER_TYPES = ["Lowpass", "Highpass", "Notch", "Bandpass"]

# Define the specific frequencies the user can choose *for Notch/Band filters*
# These are the options presented to the user if they guess Notch or Bandpass
NOTCH_BAND_CHOICES = {
    500: "Low",
    600: "Low-Mid",
    1500: "Mid",
    5000: "High-Mid",
    8000: "High",
    10000: "Very High",
}
# Create a sorted list of frequencies for presenting choices to the user
NOTCH_BAND_FREQ_LIST_SORTED = sorted(list(NOTCH_BAND_CHOICES.keys()))

# Filter design parameters
BUTTER_ORDER = 5 # Order for Butterworth filters (Lowpass, Highpass)
PEAK_EQ_BW_OCT = 1.6 # Bandwidth in Octaves for Notch/Bandpass
PEAK_EQ_GAIN_DB = 9.0 # Gain in dB (+ for Bandpass, - for Notch)


# --- Audio Generation (Using the corrected pink noise generator) ---

def generate_pink_noise(duration, fs, n_sources=16):
    """Generates pink noise using a generator-style Voss-McCartney algorithm."""
    samples = int(duration * fs)
    if samples <= 0:
        return np.array([])

    max_val = 1.0 / n_sources
    sources = np.zeros(n_sources)
    changing_indices = np.zeros(n_sources, dtype=int)
    pink = np.zeros(samples)
    acc = 0.0

    for i in range(samples):
        changed = (i + 1) & -(i + 1)
        lowest_set_bit_index = -1
        if changed != 0:
            # Correct way to find the index of the lowest set bit (0-based)
            lowest_set_bit_index = (changed & 0xFFFFFFFF).bit_length() - 1

        # Determine which source index to update based on the bit pattern
        # Use modulo arithmetic if the lowest set bit index exceeds the number of sources
        # Although typically n_sources is chosen like 16 or 32 so this might not be strictly necessary
        # if samples is smaller than 2**n_sources, but safer this way.
        idx_to_change = lowest_set_bit_index % n_sources

        # Update accumulator and source value
        acc -= sources[idx_to_change]
        new_val = np.random.uniform(-max_val, max_val)
        sources[idx_to_change] = new_val
        acc += new_val

        # Add a small amount of white noise for better spectral characteristics
        white = np.random.uniform(-max_val, max_val) * 0.1 # Adjusted white noise contribution
        pink[i] = acc + white

    # Post-processing: remove DC offset and normalize
    pink -= np.mean(pink)
    max_abs = np.max(np.abs(pink))
    if max_abs > 1e-9: # Avoid division by zero for silent noise
        pink /= max_abs
    else:
        print("Warning: Generated pink noise is nearly silent.")

    return pink * NOISE_AMPLITUDE


# --- Filtering Functions ---

# --- Butterworth Filters (Lowpass, Highpass) ---
def apply_lowpass(data, cutoff_hz, fs, order=BUTTER_ORDER):
    """Applies a Lowpass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    if normal_cutoff >= 1.0: normal_cutoff = 0.999 # Avoid exactly 1
    if normal_cutoff <= 0.0: normal_cutoff = 0.001 # Avoid exactly 0
    sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

def apply_highpass(data, cutoff_hz, fs, order=BUTTER_ORDER):
    """Applies a Highpass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    if normal_cutoff >= 1.0: normal_cutoff = 0.999 # Avoid exactly 1
    if normal_cutoff <= 0.0: normal_cutoff = 0.001 # Avoid exactly 0
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

# --- Peaking/Notching EQ Filter Design and Application ---
def design_peak_eq(f0, dBgain, BW_oct, fs):
    """
    Designs SOS coefficients for a Peaking/Notching EQ filter.
    Based on Audio EQ Cookbook by Robert Bristow-Johnson.
    f0: Center frequency (Hz)
    dBgain: Gain at center frequency (dB) (+ for boost, - for cut)
    BW_oct: Bandwidth in octaves
    fs: Sample rate (Hz)
    Returns: sos array (for use with sosfiltfilt)
    """
    # Clamp f0 to avoid issues near 0 and Nyquist
    f0 = max(1.0, min(f0, fs / 2.0 - 1.0))
    # Basic validation for bandwidth
    BW_oct = max(0.01, BW_oct)

    A = 10**(dBgain / 40.0)  # Amplitude ratio (note the /40 for peaking EQ)
    w0 = 2 * math.pi * f0 / fs
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)

    # Calculate alpha from octave bandwidth
    # Formula: alpha = sin(w0)*sinh( ln(2)/2 * BW * w0/sin(w0) )
    # Need to handle potential sin(w0) == 0 case (f0 = 0 or fs/2), though f0 clamping helps
    if abs(sin_w0) < 1e-9: # Avoid division by zero if f0 is very close to 0 or Nyquist
        # Fallback: Use a Q based approach if sin(w0) is too small
        # Q = sqrt(2^BW)/(2^BW - 1) # Relationship between Q and BW for peaking
        # A simpler approach for this edge case might be to slightly nudge f0
        # Or accept that the filter will have minimal effect at these extremes
        print(f"Warning: sin(w0) near zero for f0={f0}. EQ effect might be minimal. Using approx alpha.")
        # Approximate alpha based on Q definition for peaking EQ, Q relates to BW
        # Q = sqrt(2**BW_oct) / (2**BW_oct - 1) # Might be unstable calculation
        # A simpler relation: Q = 1 / (2*sinh(ln(2)/2*BW_oct))
        Q_approx = 1.0 / (2.0 * math.sinh(math.log(2.0) / 2.0 * BW_oct))
        alpha = sin_w0 / (2.0 * Q_approx) # Definition alpha = sin(w0)/(2*Q)

        if abs(alpha) < 1e-9: # If still too small, might indicate problem
             print(f"Warning: Alpha calculation resulted in near zero ({alpha}). Filter might be ineffective.")
             alpha = 1e-6 # Assign a very small value to avoid NaN/Inf later
    else:
        alpha = sin_w0 * math.sinh(math.log(2.0) / 2.0 * BW_oct * w0 / sin_w0)


    # Check if alpha is valid
    if not (0 < alpha < 1):
         print(f"Warning: Calculated alpha ({alpha}) is outside expected range (0, 1) for f0={f0}, BW={BW_oct}. Clamping.")
         # This might happen with extreme BW/f0 combinations. Clamp or adjust parameters.
         alpha = max(1e-6, min(alpha, 0.99)) # Clamp to a safe range

    # Calculate coefficients (pre-normalization)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    # Check for potential division by zero if A is very small (large negative gain)
    if abs(a0) < 1e-9:
         print(f"Warning: Denominator coefficient a0 is near zero (A={A}). Filter unstable/invalid. Returning identity.")
         # Return SOS for an identity filter (no change)
         return np.array([[1., 0., 0., 1., 0., 0.]])


    # Normalize coefficients so a0 = 1
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([a0 / a0, a1 / a0, a2 / a0]) # a becomes [1, a1/a0, a2/a0]

    # Convert to Second-Order Sections (SOS) format for stability
    # Reshape b and a to be 2D arrays as expected by tf2sos
    try:
        sos = signal.tf2sos(b, a, analog=False)
    except ValueError as e:
         print(f"Error converting TF to SOS: {e}. Parameters: f0={f0}, dBgain={dBgain}, BW={BW_oct}")
         print(f"Coefficients: b={b}, a={a}")
         # Return identity filter on error
         sos = np.array([[1., 0., 0., 1., 0., 0.]])

    return sos


def apply_biquad_filter(data, sos):
    """Applies a filter defined by SOS coefficients using zero-phase filtering."""
    if sos is None or sos.shape[0] == 0:
        print("Warning: Invalid SOS coefficients. Skipping filtering.")
        return data
    # Ensure data is float64 for sosfiltfilt robustness
    filtered_data = signal.sosfiltfilt(sos, data.astype(np.float64))
    return filtered_data


# --- Quiz Logic ---

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


    # --- User Preferences ---
    while True:
        show_filter_type_ans = input("Show correct FILTER TYPE after each guess? (y/n): ").lower()
        if show_filter_type_ans in ['y', 'n']:
            show_filter_type_ans = (show_filter_type_ans == 'y')
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    while True:
        show_details_ans = input("Show filter DETAILS (Freq, BW, Gain) after each guess? (y/n): ").lower()
        if show_details_ans in ['y', 'n']:
            show_details_ans = (show_details_ans == 'y')
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    print("\nStarting the quiz...")
    time.sleep(1)

    score = 0.0 # Use float for score to allow 0.5 points
    # Get list of frequencies specifically for Notch/Bandpass filters
    possible_notch_band_freqs = list(NOTCH_BAND_CHOICES.keys())

    for i in range(NUM_QUESTIONS):
        print(f"\n--- Question {i+1}/{NUM_QUESTIONS} ---")

        # 1. Choose Filter Type and Frequency
        filter_type = random.choice(FILTER_TYPES)
        correct_freq = None # Store the actual frequency used
        correct_label = ""
        details_str = "" # To store specific parameters for feedback

        if filter_type == "Lowpass":
            correct_freq = 5000 # Fixed frequency for Lowpass
            correct_label = INTERNAL_FILTER_FREQS[correct_freq]
            details_str = f"Cutoff: {correct_freq} Hz (Butterworth Order {BUTTER_ORDER})"
        elif filter_type == "Highpass":
            correct_freq = 150 # Fixed frequency for Highpass
            correct_label = INTERNAL_FILTER_FREQS[correct_freq]
            details_str = f"Cutoff: {correct_freq} Hz (Butterworth Order {BUTTER_ORDER})"
        else: # Notch or Bandpass (using Peaking EQ design)
            # Select a frequency *only* from the allowed Notch/Band choices
            if not possible_notch_band_freqs:
                 print("Error: No frequencies defined for Notch/Bandpass choices!")
                 sys.exit(1)

            correct_freq = random.choice(possible_notch_band_freqs)
            correct_label = NOTCH_BAND_CHOICES[correct_freq] # Get label from choices dict

            if filter_type == "Notch":
                current_gain = -PEAK_EQ_GAIN_DB
            else: # Bandpass
                current_gain = PEAK_EQ_GAIN_DB
            current_bw = PEAK_EQ_BW_OCT

            details_str = f"Center: {correct_freq} Hz ({correct_label}), BW: {current_bw:.1f} Oct, Gain: {current_gain:+.1f} dB"


        # 2. Generate Noise
        print("Generating noise...")
        sys.stdout.flush()
        original_noise = generate_pink_noise(DURATION, FS)

        # 3. Apply Filter using correct_freq
        filtered_noise = None
        sos_coeffs = None # Variable to hold SOS coeffs if needed

        if filter_type == "Lowpass":
            filtered_noise = apply_lowpass(original_noise, correct_freq, FS, BUTTER_ORDER)
        elif filter_type == "Highpass":
            filtered_noise = apply_highpass(original_noise, correct_freq, FS, BUTTER_ORDER)
        elif filter_type == "Notch":
            sos_coeffs = design_peak_eq(correct_freq, -PEAK_EQ_GAIN_DB, PEAK_EQ_BW_OCT, FS)
            filtered_noise = apply_biquad_filter(original_noise, sos_coeffs)
        elif filter_type == "Bandpass":
            sos_coeffs = design_peak_eq(correct_freq, PEAK_EQ_GAIN_DB, PEAK_EQ_BW_OCT, FS)
            filtered_noise = apply_biquad_filter(original_noise, sos_coeffs)

        # Ensure filtered noise is not silent or invalid
        if filtered_noise is None or np.max(np.abs(filtered_noise)) < 1e-6:
             print("Warning: Filter resulted in near-silent audio. Using original noise for filtered playback.")
             filtered_noise = original_noise # Fallback

        # Optional: RMS Normalization (can sometimes make EQ effects less obvious)
        # original_rms = np.sqrt(np.mean(original_noise**2))
        # filtered_rms = np.sqrt(np.mean(filtered_noise**2))
        # if filtered_rms > 1e-9:
        #    # Normalize filtered to match original RMS
        #    filtered_noise *= (original_rms / filtered_rms)
        # # Ensure clipping doesn't occur after normalization
        # filtered_noise = np.clip(filtered_noise, -NOISE_AMPLITUDE, NOISE_AMPLITUDE)


        # 4. Play Sounds
        try:
            print("Playing ORIGINAL noise...")
            sys.stdout.flush()
            sd.play(original_noise, FS)
            sd.wait()
            time.sleep(PAUSE_BETWEEN_SOUNDS)

            print("Playing FILTERED noise...")
            sys.stdout.flush()
            sd.play(filtered_noise, FS)
            sd.wait()

        except Exception as e:
            print(f"\nError during audio playback: {e}")
            print("Please ensure you have a working audio output device and 'sounddevice' is installed correctly.")
            print("You might need to install PortAudio ('sudo apt-get install portaudio19-dev' on Debian/Ubuntu, 'brew install portaudio' on macOS).")
            sys.exit(1)


        # 5. Get User Guess (Type and maybe Frequency)
        user_guess_type = None
        user_guess_freq = None

        # -- Get Filter Type Guess --
        while True:
            raw_guess = input(f"Guess the filter type ({', '.join(FILTER_TYPES)}): ").strip().lower()
            # Allow abbreviations
            if raw_guess in ['lp', 'low', 'lowpass']: user_guess_type = 'Lowpass'
            elif raw_guess in ['hp', 'high', 'highpass']: user_guess_type = 'Highpass'
            elif raw_guess in ['n', 'notch']: user_guess_type = 'Notch'
            elif raw_guess in ['bp', 'band', 'bandpass']: user_guess_type = 'Bandpass'
            else: user_guess_type = raw_guess.capitalize() # Default attempt

            if user_guess_type in FILTER_TYPES:
                break
            else:
                print(f"Invalid guess '{raw_guess}'. Please enter one of: {', '.join(FILTER_TYPES)} (or abbreviations like lp, hp, n, bp)")

        # -- Get Frequency Guess (Only if user guessed Notch or Bandpass) --
        if user_guess_type in ["Notch", "Bandpass"]:
            # Prepare frequency choice string
            choice_strs = [f"{f} Hz ({NOTCH_BAND_CHOICES[f]})" for f in NOTCH_BAND_FREQ_LIST_SORTED]
            prompt_choices = ", ".join(choice_strs)
            print(f"Available frequencies: {prompt_choices}")

            while True:
                freq_guess_str = input(f"Guess the center frequency (enter just the number, e.g., {NOTCH_BAND_FREQ_LIST_SORTED[0]}): ").strip()
                try:
                    user_guess_freq = int(freq_guess_str)
                    if user_guess_freq in NOTCH_BAND_FREQ_LIST_SORTED:
                        break
                    else:
                        print(f"Invalid frequency. Please enter one of the listed numbers: {NOTCH_BAND_FREQ_LIST_SORTED}")
                except ValueError:
                    print("Invalid input. Please enter a number.")

        # 6. Check Answer and Give Feedback
        current_question_score = 0.0
        correct_type = (user_guess_type == filter_type)
        correct_freq_guess = False # Assume false initially

        result_message = "Incorrect." # Default message

        if filter_type in ["Lowpass", "Highpass"]:
            if correct_type:
                current_question_score = 1.0
                result_message = "Correct!"
        elif filter_type in ["Notch", "Bandpass"]:
            if correct_type:
                # User guessed the correct type, now check frequency
                if user_guess_freq == correct_freq:
                    current_question_score = 1.0
                    result_message = "Correct! (Type and Frequency)"
                    correct_freq_guess = True # Mark frequency guess as correct
                else:
                    current_question_score = 0.5
                    result_message = f"Partially Correct. (Correct type '{filter_type}', but wrong frequency)"
            else:
                # User guessed the wrong type entirely
                current_question_score = 0.0
                result_message = f"Incorrect type."
                # No points awarded if type is wrong

        print(result_message)
        score += current_question_score

        # Construct detailed feedback
        feedback = []
        if show_filter_type_ans:
            if not correct_type:
                 feedback.append(f"The correct filter type was: {filter_type}")
            # If Notch/Band and type was correct but freq wrong, or if type was wrong
            if filter_type in ["Notch", "Bandpass"] and not correct_freq_guess and user_guess_type == filter_type:
                 feedback.append(f"The correct frequency was: {correct_freq} Hz ({correct_label})")

        # Show full details if requested, regardless of correctness (useful for learning)
        if show_details_ans:
             # Ensure details are shown even if type/freq feedback was already added
             detail_prefix = "Filter Details: "
             if feedback and "Details:" in feedback[-1]: # Avoid duplicate prefix
                 feedback[-1] = f"{feedback[-1].replace('Filter Details: ','')} | {details_str}"
             elif not any("Details:" in f for f in feedback):
                 feedback.append(f"{detail_prefix}{details_str}")

        if feedback:
            print(" | ".join(feedback))

        # Display score with potential .5
        print(f"Current Score: {score:.1f}/{i+1}")
        time.sleep(PAUSE_BETWEEN_QUESTIONS)


    # --- End of Quiz ---
    print("\n" + "=" * 30)
    print("      Quiz Complete!")
    print(f"      Final Score: {score:.1f}/{NUM_QUESTIONS}") # Format score nicely
    print("=" * 30)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Check if sounddevice can find devices before starting
        try:
             print("Available audio output devices:", sd.query_devices(kind='output'))
             # Attempt a short silent playback to test device readiness
             sd.play(np.zeros(10), FS, blocking=True)
        except Exception as e:
             print("\n----- Audio Device Warning -----")
             print(f"Error initializing audio device: {e}")
             print("Please ensure a default audio output device is selected and working.")
             print("Common solutions:")
             print(" - Check system audio settings.")
             print(" - Ensure speakers/headphones are connected and powered on.")
             print(" - On Linux, you may need 'portaudio19-dev' (Debian/Ubuntu) or similar.")
             print(" - On macOS, ensure output isn't muted or set to an unavailable device.")
             print("Attempting to continue, but playback might fail.")
             print("--------------------------------\n")
             time.sleep(3) # Give user time to read

        run_quiz()
    except KeyboardInterrupt:
        print("\nQuiz interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the quiz: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        print("Exiting.")
    finally:
        # Ensure sound playback is stopped if interrupted.
        sd.stop()