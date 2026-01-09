from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import librosa
from scipy.fft import fft
from scipy.signal.windows import hamming
from scipy.signal import find_peaks
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

SAMPLE_RATE = 16000
N_MFCC = 20
MODEL_PATH = 'vocal_model.joblib'
SCALER_PATH = 'vocal_scaler.joblib'

class VocalClassifier:
    def __init__(self):
        self.scaler = None
        self.svm = None
        self.is_trained = False
        
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.load_model()

    def extract_features(self, audio, sr):
        if len(audio) < 2048:
            return None

        windowed = audio * hamming(len(audio))
        fft_vals = np.abs(fft(windowed))[:len(windowed)//2]
        freqs = np.fft.fftfreq(len(windowed), 1/sr)[:len(fft_vals)]

        features = []

        if np.sum(fft_vals) > 0:
            centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_vals) / np.sum(fft_vals))
            rolloff = freqs[np.searchsorted(np.cumsum(fft_vals), 0.85 * np.sum(fft_vals))]
        else:
            centroid = bandwidth = rolloff = 0

        features.extend([centroid, bandwidth, rolloff])

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        peaks, _ = find_peaks(fft_vals, height=np.max(fft_vals)*0.1, distance=20)
        formants = sorted(freqs[peaks], reverse=True)[:5]
        features.extend(formants + [0]*(10-len(formants)))

        zcr = np.mean(librosa.zero_crossings(audio))
        rms = np.sqrt(np.mean(audio**2))
        flatness = np.exp(np.mean(np.log(fft_vals + 1e-9))) / (np.mean(fft_vals) + 1e-9)

        features.extend([zcr, rms, flatness])

        return np.array(features)

    def load_model(self):
        try:
            self.svm = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.is_trained = True
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.is_trained = False

    def predict(self, audio, sr):
        if not self.is_trained:
            return None, 0, "Modelo no disponible"

        feats = self.extract_features(audio, sr)
        if feats is None:
            return None, 0, "Audio muy corto"

        feats = self.scaler.transform([feats])
        pred = self.svm.predict(feats)[0]
        conf = np.max(self.svm.predict_proba(feats))

        return pred, conf, None

classifier = VocalClassifier()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/status')
def status():
    return jsonify({
        'is_trained': classifier.is_trained
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'Sin audio'}), 400

    audio_bytes = request.files['audio'].read()
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    audio, _ = librosa.effects.trim(audio)

    vocal, conf, err = classifier.predict(audio, SAMPLE_RATE)
    if err:
        return jsonify({'error': err}), 400

    return jsonify({'vocal': vocal, 'confidence': conf})

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" DETECTOR DE VOCALES EN TIEMPO REAL")
    print("="*60)
    print("URL: http://localhost:5000")
    print(f"Estado: {'Modelo cargado' if classifier.is_trained else 'Modelo no encontrado'}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)