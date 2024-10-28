import cv2
import time
import pandas as pd
import random
from collections import Counter
from deepface import DeepFace
import urllib.parse

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-model.yml") 
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

names = ['None', 'Yujin']  # Change Into Your Name

songs_df = pd.read_csv('songs_normalize.csv')

def generate_spotify_link(song, artist):
    base_url = "https://open.spotify.com/search/"
    query = f"{song} {artist}"
    encoded_query = urllib.parse.quote(query)
    return base_url + encoded_query

def recommend_songs(emotion, num_songs=5):
    if emotion == 'happy':
        recommendations = songs_df[
            (songs_df['valence'] > 0.6) & (songs_df['energy'] > 0.5) & (songs_df['danceability'] > 0.5) & (songs_df['mode'] == 1)
        ]
    elif emotion == 'sad':
        recommendations = songs_df[
            (songs_df['valence'] < 0.4) & (songs_df['energy'] < 0.4) & (songs_df['danceability'] < 0.5) & (songs_df['mode'] == 0)
        ]
    elif emotion == 'fear':
        recommendations = songs_df[
            (songs_df['valence'] < 0.4) & (songs_df['energy'] > 0.5) & (songs_df['loudness'] > -10) & (songs_df['tempo'] > 100)
        ]
    elif emotion == 'angry':
        recommendations = songs_df[
            (songs_df['valence'] < 0.5) & (songs_df['energy'] > 0.7) & (songs_df['loudness'] > -8) & (songs_df['tempo'] > 120)
        ]
    elif emotion == 'surprised':
        recommendations = songs_df[
            (songs_df['energy'] > 0.6) & (songs_df['tempo'] > 110) & (songs_df['valence'] > 0.5)
        ]
    elif emotion == 'neutral':
        recommendations = songs_df.sample(num_songs)
    else:
        print("Emotion not recognized.")
        return []

    recommendations = recommendations.drop_duplicates(subset=['song', 'artist'])

    if not recommendations.empty:
        recommendations = recommendations.sample(min(num_songs, len(recommendations)))
        recommendations['spotify_link'] = recommendations.apply(
            lambda row: generate_spotify_link(row['song'], row['artist']), axis=1
        )
        return recommendations
    else:
        print("No songs match the criteria for this emotion.")
        return []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

emotion_counts = Counter()
start_time = time.time()
scan_duration = 5 # in Seconds (please set it up based on ur machine's capabilities)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #On Screen Countdown
    elapsed_time = time.time() - start_time
    remaining_time = max(0, int(scan_duration - elapsed_time))

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        name = names[id] if confidence < 100 else "Unknown"
        
        try:
            analysis = DeepFace.analyze(frame[y:y + h, x:x + w], actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            dominant_emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
            emotion_counts[dominant_emotion] += 1
            
        except Exception as e:
            print("Emotion analysis error:", e)
            dominant_emotion = "N/A"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Name: {name}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Confidence: {int(100 - confidence)}%", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    cv2.putText(frame, f"Scanning... {remaining_time}s left", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Camera", frame)

    if elapsed_time > scan_duration or cv2.waitKey(1) == ord("q"):
        break


if emotion_counts:
    most_dominant_emotion = emotion_counts.most_common(1)[0][0]
    print("Most Dominant Emotion:", most_dominant_emotion)
    
    recommended_songs = recommend_songs(most_dominant_emotion)
    
    if not recommended_songs.empty:
        print("Recommended songs based on emotion:")
        for _, row in recommended_songs.iterrows():
            print(f"{row['artist']} - {row['song']} ({row['genre']}, {row['year']})")
            print("Listen on Spotify:", row['spotify_link'])
            print()
else:
    print("No dominant emotion detected.")

cap.release()
cv2.destroyAllWindows()
