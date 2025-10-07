import os
import numpy as np
import librosa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, Layer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. Parametreler
DATA_PATH = r"C:\Users\kdrt2\OneDrive\Masaüstü\emotion-recognition-app\emotion-recognition-app1\modeller\Audio_Speech_Actors_01-24 (1)"
n_mfcc = 40

# 2. Veri yükleme
X, y = [], []
for label in os.listdir(DATA_PATH):
    class_path = os.path.join(DATA_PATH, label)
    if not os.path.isdir(class_path):
        continue
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)
        try:
            signal, sr = librosa.load(file_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T  # (time, n_mfcc)
            X.append(mfcc)
            y.append(label)
        except Exception as e:
            print(f"Hata: {file_path} -> {e}")

# 3. Padding (her örneği aynı uzunluğa getirmek için)
max_len = max([mfcc.shape[0] for mfcc in X])
X_pad = np.array([np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant') for mfcc in X])

# 4. Etiket encode
le = LabelEncoder()
y_enc = to_categorical(le.fit_transform(y))

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_enc, test_size=0.2, random_state=42)

# 6. Transformer Bloğu
class TransformerBlock(Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.ffn_dense1 = Dense(ff_dim, activation="relu")
        self.ffn_dropout = Dropout(dropout)
        self.ffn_dense2 = None  # build içinde ayarlanacak
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        feature_dim = input_shape[-1]  # örn. 40 mfcc
        self.ffn_dense2 = Dense(feature_dim)

    def call(self, inputs, training=False):
        # Multi-head attention + skip connection
        attn_out = self.att(inputs, inputs)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(inputs + attn_out)

        # Feed Forward + skip connection
        ffn_out = self.ffn_dense1(x)
        ffn_out = self.ffn_dropout(ffn_out, training=training)
        ffn_out = self.ffn_dense2(ffn_out)
        return self.norm2(x + ffn_out)

# 7. Model
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = TransformerBlock(64, 4, 128)(inputs)
x = TransformerBlock(64, 4, 128)(x)

x = GlobalAveragePooling1D()(x)
x = Dense(128, activation="relu")(x)  # ekstra katman
x = Dropout(0.5)(x)
outputs = Dense(y_train.shape[1], activation="softmax")(x)

model = Model(inputs, outputs)

# 8. Compile
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001),
              metrics=["accuracy"])

# 9. Train (EarlyStopping ile)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

# 10. Kaydet
model.save("transformer_model.h5")

print("✅ Transformer modeli başarıyla eğitildi ve kaydedildi!")
