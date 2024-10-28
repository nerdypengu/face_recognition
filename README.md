Here's a GitHub-friendly README for your project:

---

# Emotion-Based Music Recommendation System 🎶😊

This project combines **facial emotion detection** with **music recommendation**. Using a webcam, the system detects a user’s emotion and then suggests music tailored to that emotion from a Spotify dataset.

## Features

- **Real-Time Emotion Detection**: Uses a webcam feed to detect emotions in real-time with [DeepFace](https://github.com/serengil/deepface).
- **Music Recommendation**: Based on the detected emotion, recommends songs with relevant features (e.g., high valence for happy emotions).
- **Spotify Integration**: Generates a Spotify search link to easily find and listen to the recommended songs.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Code Explanation](#code-explanation)
6. [Contributing](#contributing)

---

## Getting Started

This project requires:
- A **webcam** for real-time video capture.
- A **trained face recognizer model** (LBPH) and a **Haar Cascade classifier**.
- A **Spotify music dataset** to use for recommendations. 

### Prerequisites

- Python 3.x
- Kaggle account to download the Spotify music dataset

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-music-recommender.git
   cd emotion-music-recommender
   ```

2. **Install the required libraries:**
   ```bash
   pip install opencv-python-headless deepface pandas
   ```

3. **Download the Spotify dataset from Kaggle** and place it in the project directory:
   - [Spotify Dataset on Kaggle](https://www.kaggle.com/datasets/abdelrahman16/spotify-analysis-and-visualization)

4. **Prepare the face model:**
   - Train your face recognizer model (LBPH) or use an existing one, saving it as `face-model.yml`.
   - Ensure that the `haarcascade_frontalface_default.xml` file is available for face detection.

---

## Usage

To start the application:
```bash
python emotion_music_recommender.py
```

1. **Emotion Detection**: The system scans the face for 8 seconds, then identifies the most frequently detected emotion.
2. **Music Recommendation**: Based on the detected emotion, the system suggests five songs and generates Spotify links.

---

## Dataset

The dataset is a **Spotify music dataset** containing features like `valence`, `energy`, `danceability`, etc., essential for emotion-based recommendations. The dataset can be downloaded from Kaggle:

[Spotify Dataset - Kaggle](https://www.kaggle.com/datasets/abdelrahman16/spotify-analysis-and-visualization)

Ensure the file is named `songs_normalize.csv`.

---

## Code Explanation

### Face Recognition and Emotion Detection

The script uses **OpenCV** and **DeepFace** to recognize faces and detect emotions. The program captures video frames, processes them for faces, and then analyzes emotions based on detected faces.

```python
# Set up camera capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Emotion Counter

The system captures and counts emotions over 8 seconds to determine the most common emotion detected.

```python
from collections import Counter
emotion_counts = Counter()
scan_duration = 8  # seconds
```

### Song Recommendation

Based on the dominant emotion detected, the system filters songs from the dataset with matching features.

```python
# Recommend songs function
def recommend_songs(emotion, num_songs=5):
    # Filters for happy, sad, fear, angry, surprised, and neutral emotions
    # Each emotion corresponds to specific song features
```

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

---

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- [Kaggle Spotify Dataset](https://www.kaggle.com/datasets/abdelrahman16/spotify-analysis-and-visualization)

