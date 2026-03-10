# src/models.py
"""
Load và xử lý BERT models
"""

import os
# Fix TensorFlow locking issue on macOS
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel


from .rules import DATA_PATH, BERT_MODEL_DIR, BERT_MAX_LENGTH


class BERTModel:
    """Class quản lý BERT model và data"""
    
    def __init__(self):
        self.data = None
        self.bert_embeddings = None
        self.cosine_sim = None
        self.tokenizer = None
        self.encoder = None
        self.device = None
        self._bert_loaded = False

        
        # Load data ngay (không cần TensorFlow)
        self._load_data()
        self._compute_cosine_similarity()
    
    def _load_data(self):
        """Load data từ pickle file"""
        print("Đang load dữ liệu...")
        self.data = pd.read_pickle(DATA_PATH)
        self.bert_embeddings = np.vstack(self.data['bert_vector'].to_numpy())
        print(f"✅ Đã load {len(self.data)} bài tập")
    
    def _load_bert_model(self):
        """Load HuggingFace tokenizer + encoder (chỉ khi cần)"""
        if self._bert_loaded:
            return

        print("Đang load BERT model (HuggingFace)...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
            self.encoder = AutoModel.from_pretrained(BERT_MODEL_DIR).to(self.device).eval()

            self._bert_loaded = True
            print("✅ BERT model đã sẵn sàng")
        except Exception as e:
            print(f"❌ Không thể load BERT model: {e}")
            raise


    
    def _compute_cosine_similarity(self):
        """Tính toán ma trận cosine similarity cho tất cả bài tập"""
        print("Đang tính toán similarity matrix...")
        self.cosine_sim = cosine_similarity(self.bert_embeddings, self.bert_embeddings)
        print("✅ Similarity matrix đã sẵn sàng")
    
def get_bert_vector_for_text(self, text):
    """
    Chuyển đổi text thành BERT pooled_output vector bằng model local của bạn
    Returns: numpy array shape (1, hidden_size)
    """
    if not self._bert_loaded:
        self._load_bert_model()

    import torch
    with torch.no_grad():
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=BERT_MAX_LENGTH,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.encoder(**inputs)
        return outputs.pooler_output.cpu().numpy()

    
    def get_data(self):
        """Trả về dataframe"""
        return self.data
    
    def get_embeddings(self):
        """Trả về BERT embeddings"""
        return self.bert_embeddings
    
    def get_cosine_sim(self):
        """Trả về ma trận cosine similarity"""
        return self.cosine_sim


# Singleton instance
_model_instance = None

def get_model():
    """
    Lấy instance của BERTModel (singleton pattern)
    
    Returns:
        BERTModel: Instance duy nhất của model
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = BERTModel()
    return _model_instance