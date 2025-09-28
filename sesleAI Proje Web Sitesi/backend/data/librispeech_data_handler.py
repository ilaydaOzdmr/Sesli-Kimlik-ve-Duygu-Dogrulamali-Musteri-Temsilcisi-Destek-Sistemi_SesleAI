import os
import glob
from collections import defaultdict
import random

class LibriSpeechDataHandler:
    def __init__(self, data_path):
        """
        LibriSpeech veri setini işlemek için yardımcı sınıf.

        Args:
            data_path (str): LibriSpeech veri setinin ana klasör yolu.
                             (örn: 'backend/data/')
        """
        self.data_path = data_path
        self.speaker_ids = self._load_speaker_ids()
        self.all_audio_files = self._get_all_audio_files()

    def _load_speaker_ids(self):
        """
        SPEAKERS.TXT dosyasından konuşmacı ID'lerini yükler.
        """
        speakers_file = os.path.join(self.data_path, "SPEAKERS.TXT")
        speaker_ids = {}
        with open(speakers_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith(';'):
                    parts = line.split('|')
                    speaker_id = parts[0].strip()
                    gender = parts[1].strip()
                    subset = parts[2].strip()
                    name = parts[3].strip()
                    speaker_ids[speaker_id] = {'gender': gender, 'subset': subset, 'name': name}
        return speaker_ids

    def _get_all_audio_files(self):
        """
        Veri setindeki tüm .flac ses dosyalarının yollarını bulur.
        """
        audio_files = []
        # 'dev-clean', 'test-clean', 'train-clean-100' gibi klasörleri arar
        for subset in os.listdir(self.data_path):
            subset_path = os.path.join(self.data_path, subset)
            if os.path.isdir(subset_path):
                # Tüm .flac dosyalarını bulmak için glob kullanır
                audio_files.extend(glob.glob(os.path.join(subset_path, '**/*.flac'), recursive=True))
        return audio_files

    def get_data_for_training(self, max_speakers=None, max_utterances_per_speaker=None):
        """
        Model eğitimi için veri hazırlığı yapar (ses dosyaları ve etiketler).

        Args:
            max_speakers (int, optional): Kullanılacak maksimum konuşmacı sayısı.
            max_utterances_per_speaker (int, optional): Her konuşmacıdan alınacak maksimum ses kaydı sayısı.

        Returns:
            dict: Konuşmacı ID'lerine göre gruplanmış ses dosyalarının sözlüğü.
        """
        speaker_data = defaultdict(list)
        for file_path in self.all_audio_files:
            # Dosya yolundan konuşmacı ID'sini çıkarır
            # Örneğin: .../19/198/19-198-0000.flac -> '19'
            speaker_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            
            # Konuşmacının SPEAKERS.TXT dosyasında olup olmadığını kontrol et
            if speaker_id in self.speaker_ids:
                speaker_data[speaker_id].append(file_path)

        # Eğer maksimum konuşmacı sayısı belirtildiyse, rastgele seçim yap
        if max_speakers:
            selected_speakers = random.sample(list(speaker_data.keys()), min(max_speakers, len(speaker_data)))
            speaker_data = {speaker: speaker_data[speaker] for speaker in selected_speakers}

        # Eğer her konuşmacıdan maksimum ses kaydı belirtildiyse, kırp
        if max_utterances_per_speaker:
            for speaker_id in speaker_data:
                random.shuffle(speaker_data[speaker_id])
                speaker_data[speaker_id] = speaker_data[speaker_id][:max_utterances_per_speaker]

        return speaker_data

if __name__ == '__main__':
    # Bu kısım, dosyanın nasıl çalıştığını test etmek için
    # backend/data klasörünün yolunu belirtin.
    base_data_path = 'data' 
    handler = LibriSpeechDataHandler(base_data_path)
    
    # Tüm verileri yükle (dikkat: bu çok fazla olabilir)
    # all_data = handler.get_data_for_training()
    # print(f"Toplam {len(all_data)} konuşmacı bulundu.")

    # Sadece 10 konuşmacıdan 5'er ses dosyası alarak test et
    sample_data = handler.get_data_for_training(max_speakers=10, max_utterances_per_speaker=5)
    print(f"Eğitim için {len(sample_data)} konuşmacıdan örnek veri alındı.")
    
    for speaker_id, files in sample_data.items():
        print(f"Konuşmacı ID: {speaker_id}, Dosya Sayısı: {len(files)}")
        # print("Örnek dosya:", files[0])