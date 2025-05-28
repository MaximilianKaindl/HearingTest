# 🎧 Hearing Test Quiz - Train Your Ear to Identify Audio Filters

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Sharpen your audio perception and enhance your mixing, mastering, and sound design skills with this interactive web-based hearing test quiz focused on common audio filters!

## ✨ What is it?

This web application is designed to help you train your ear to recognize the subtle (and sometimes not-so-subtle!) effects of common audio filters. Whether you're an aspiring audio engineer, a seasoned producer, or just curious about how filters shape sound, this quiz provides a structured way to test and improve your listening abilities.

The core loop is simple: you hear an original sound source, then you hear the same sound source with a filter applied. Your task is to identify the type of filter (Lowpass, Highpass, Notch, or Bandpass) and, for Notch and Bandpass filters, estimate the center frequency from a set of common frequency ranges.

## 🚀 Features

*   **Interactive Quiz:** Test your ear across 10 challenging questions.
*   **Learn by Doing:** Immediate feedback shows you if your guess was correct and provides the actual filter details.
*   **Multiple Filter Types:** Practice identifying Lowpass, Highpass, Notch, and Bandpass filters.
*   **Common Frequency Ranges:** Focus on recognizing filters centered around typical frequencies used in audio production (100 Hz, 600 Hz, 1500 Hz, 5000 Hz, 8000 Hz, 10000 Hz).
*   **Customizable Audio Source:**
    *   Use built-in **Pink Noise** for a broadband, consistent signal that makes filter effects very clear.
    *   **Upload Your Own Audio File** (MP3, WAV, OGG, M4A) to train with sounds you are familiar with or typically work with.
*   **Adjustable Sample Duration:** Choose how long the audio samples play for (1s, 2s, 3s, 5s).
*   **Focus Mode:** Optionally isolate your training to only Notch and Bandpass filters, focusing purely on frequency identification.
*   **Score Tracking:** See your score update after each question and view your final result.
*   **Example Player:** Familiarize yourself with the sound of each filter type and frequency before taking the quiz, using your chosen audio source and duration.
*   **Single File Simplicity:** The entire application is contained within one HTML file, making it incredibly easy to set up and run.
*   **Pure Web Technology:** Built with HTML, CSS, and JavaScript using the Web Audio API for high-quality audio processing directly in the browser.

## 🎮 How to Use

1.  **Open:** Simply open the `index.html` file in any modern web browser (like Chrome, Firefox, Safari, Edge). No server or installation is required!
2.  **Adjust Settings:** On the setup page, customize your quiz experience:
    *   Choose the **Audio Sample Duration** (how long each sound plays).
    *   Select the **Audio Source** (Pink Noise or upload your file). If uploading a file, ensure it's long enough for the selected duration.
    *   Choose the **Filter Mode** (Random types or Focus on Notch/Bandpass).
    *   Enable **Options** like showing correct answers or filter details after each guess.
3.  **Explore Examples (Recommended!):** Use the "Hear Filter Examples" section to select a filter type and frequency and click "Play Example" to hear how it sounds with your chosen audio source and duration. This is a great way to calibrate your ears before the quiz.
4.  **Start Quiz:** Click the "Start Quiz" button when you're ready.
5.  **Play & Guess:**
    *   Click "▶️ Play Original" to hear the unfiltered sound.
    *   Click "▶️ Play Filtered" to hear the sound with the filter applied.
    *   Select the filter type and, if needed, the center frequency from the options.
    *   Click "Submit Answer".
6.  **Feedback:** See your feedback and score, then click "Next Question".
7.  **Finish:** Complete all 10 questions to see your final score. Click "Take Quiz Again" to restart.

**Note:** Using headphones is highly recommended for accurate frequency perception. Adjust your system volume to a comfortable level before starting.

## 🛠️ Local Setup

This project is designed for maximum simplicity.

1.  **Download:** Download the `index.html` file (and the README.md if you want this file).
2.  **Open:** Open the `index.html` file directly in your web browser.