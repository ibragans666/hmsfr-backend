from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import random

# Definisikan SCALES (digunakan untuk validasi)
SCALES = {
    'C': {'major': ['C', 'Dm', 'Em', 'F', 'G', 'Am'], 'minor': ['Cm', 'Eb', 'Fm', 'Gm', 'Ab', 'Bb']},
    'D': {'major': ['D', 'Em', 'F#m', 'G', 'A', 'Bm'], 'minor': ['Dm', 'F', 'Gm', 'Am', 'Bb', 'C']},
    'E': {'major': ['E', 'F#m', 'G#m', 'A', 'B', 'C#m'], 'minor': ['Em', 'G', 'Am', 'Bm', 'C', 'D']},
    'F': {'major': ['F', 'Gm', 'Am', 'Bb', 'C', 'Dm'], 'minor': ['Fm', 'Ab', 'Bbm', 'Cm', 'Db', 'Eb']},
    'G': {'major': ['G', 'Am', 'Bm', 'C', 'D', 'Em'], 'minor': ['Gm', 'Bb', 'Cm', 'Dm', 'Eb', 'F']},
    'A': {'major': ['A', 'Bm', 'C#m', 'D', 'E', 'F#m'], 'minor': ['Am', 'C', 'Dm', 'Em', 'F', 'G']},
    'B': {'major': ['B', 'C#m', 'D#m', 'E', 'F#', 'G#m'], 'minor': ['Bm', 'D', 'Em', 'F#m', 'G', 'A']}
}

# Inisialisasi FastAPI
app = FastAPI()

# Tambahkan CORS agar Next.js bisa mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain Next.js setelah deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder untuk predict_sentiment (Model 1)
def predict_sentiment(line, sentiment_model):
    if "cinta" in line.lower():
        return 1
    if "sepi" in line.lower():
        return 0
    return 1

# Placeholder untuk Model 2 (ChordProgressionTransformer)
class DummyModel:
    def __init__(self):
        pass

    def load_state_dict(self, path):
        pass

    def eval(self):
        pass

    def __call__(self, tokenized, sent, length):
        return torch.tensor([[random.randint(0, 5) for _ in range(length.item())]])

model = DummyModel()
sentiment_model = None

def tokenize_text(text):
    return [0] * len(text.split())

chord_to_idx = {'C': 0, 'Dm': 1, 'Em': 2, 'F': 3, 'G': 4, 'Am': 5}
idx_to_chord = {v: k for k, v in chord_to_idx.items()}

device = torch.device('cpu')

# Fungsi Model 2: generate_lyrics_and_progression
def generate_lyrics_and_progression(lyrics, user_root='C'):
    segments = ['verse', 'prechorus', 'chorus']
    max_lines_per_segment = 6
    min_lines_per_segment = 2

    sentiments = {}
    for segment in segments:
        sentiments[segment] = []
        for line in lyrics[segment]:
            if line:
                sent = predict_sentiment(line, sentiment_model)
            else:
                sent = 0
            sentiments[segment].append(sent)

    if user_root not in SCALES:
        user_root = 'C'
    root = user_root
    root_idx = chord_to_idx.get(root, 0)

    progressions = {}
    progression_lengths = {}
    model.load_state_dict('models/chord_progression_transformer.pt')
    model.eval()

    for segment in segments:
        valid_lines = [line for line in lyrics[segment] if line]
        num_lines = len(valid_lines)
        total_words = sum(len(line.split()) for line in valid_lines)
        prog_length = min(max(2, total_words // 3), 6)
        progression_lengths[segment] = prog_length

        seg_sentiments = sentiments[segment][:num_lines]
        majority_sentiment = 1 if sum(seg_sentiments) > len(seg_sentiments) / 2 else 0 if seg_sentiments else 0
        positive_count = sum(1 for s in seg_sentiments if s == 1)
        if positive_count >= 4 and majority_sentiment == 0 and len(seg_sentiments) >= 4:
            majority_sentiment = 1

        scale_type = 'major' if majority_sentiment == 1 else 'minor'
        allowed_chords = SCALES[root][scale_type]

        valid_lyrics = ' '.join(valid_lines)
        tokenized = torch.tensor([tokenize_text(valid_lyrics)], dtype=torch.long).to(device)
        sent = torch.tensor([majority_sentiment], dtype=torch.long).to(device)
        length = torch.tensor([prog_length], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(tokenized, sent, length)
            preds = torch.argmax(output, dim=2).cpu().numpy()[0][:prog_length]
            predicted_chords = [idx_to_chord.get(idx, root) for idx in preds]

        adjusted_progression = []
        used_chords = set()
        for chord in predicted_chords:
            if chord in allowed_chords:
                if chord not in used_chords:
                    adjusted_progression.append(chord)
                    used_chords.add(chord)
                else:
                    available_chords = [c for c in allowed_chords if c not in used_chords]
                    if available_chords:
                        next_chord = random.choice(available_chords)
                        adjusted_progression.append(next_chord)
                        used_chords.add(next_chord)
                    else:
                        adjusted_progression.append(root)
            else:
                base_note = chord.split('m')[0].split('7')[0].split('maj')[0]
                for allowed in allowed_chords:
                    if allowed.startswith(base_note) and allowed not in used_chords:
                        adjusted_progression.append(allowed)
                        used_chords.add(allowed)
                        break
                else:
                    available_chords = [c for c in allowed_chords if c not in used_chords]
                    if available_chords:
                        next_chord = random.choice(available_chords)
                        adjusted_progression.append(next_chord)
                        used_chords.add(next_chord)
                    else:
                        adjusted_progression.append(root)

        while len(adjusted_progression) < prog_length:
            available_chords = [c for c in allowed_chords if c not in used_chords]
            if available_chords:
                next_chord = random.choice(available_chords)
                adjusted_progression.append(next_chord)
                used_chords.add(next_chord)
            else:
                adjusted_progression.append(root)
        progressions[segment] = adjusted_progression[:prog_length]

    return lyrics, sentiments, progressions, progression_lengths, root

# Endpoint API untuk generate progresi awal
@app.post("/generate-progression")
async def generate_progression(request: Request):
    data = await request.json()
    lyrics = data['lyrics']
    user_root = data['root']

    lyrics, sentiments, progressions, progression_lengths, root = generate_lyrics_and_progression(lyrics, user_root)

    return {
        "lyrics": lyrics,
        "sentiments": sentiments,
        "progressions": progressions,
        "progression_lengths": progression_lengths,
        "root": root
    }