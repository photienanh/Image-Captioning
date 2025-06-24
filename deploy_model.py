import torch
import pickle
from model import Encoder, Decoder
from PIL import Image
from torchvision import transforms

def load_model_and_utils(model_path='Models/caption_model.pth'):
    # Load word mappings
    with open('Processed Data/word_to_index.pkl', 'rb') as f:
        word_to_index = pickle.load(f)
    with open('Processed Data/index_to_word.pkl', 'rb') as f:
        index_to_word = pickle.load(f)
    with open('Processed Data/embedding_matrix.pkl', 'rb') as f:
        embedding_matrix = pickle.load(f)

    vocab_size = len(word_to_index)
    embedding_dim = embedding_matrix.shape[1]
    hidden_dim = 256

    encoder = Encoder(encoded_image_size=hidden_dim)
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        embedding_matrix=embedding_matrix
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return encoder, decoder, word_to_index, index_to_word, transform, device

def extract_feature(img, encoder, transform, device):
    # img: PIL Image
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    with torch.no_grad():
        feat = encoder.resnet(img_tensor)  # (1, 2048, 1, 1)
        feature = feat.view(feat.size(0), -1)  # (1, 2048)
    return feature

def generate_caption(feature, encoder, decoder, word_to_index, index_to_word, max_len=37, device='cpu'):
    with torch.no_grad():
        feature_encoded = encoder.linear(encoder.dropout(feature)).to(device)  # (1, 256)
        caption = [word_to_index['<START>']]
        for _ in range(max_len):
            cap_tensor = torch.tensor([caption], dtype=torch.long).to(device)  # (1, seq_len)
            outputs = decoder(cap_tensor, feature_encoded)  # (1, seq_len, vocab_size)
            next_word_logits = outputs[0, -1, :]  # (vocab_size,)
            next_word = next_word_logits.argmax().item()
            caption.append(next_word)
            if next_word == word_to_index['<END>']:
                break
        words = [index_to_word[idx] for idx in caption[1:]]
        if '<END>' in words:
            words = words[:words.index('<END>')]
        return ' '.join(words)

def predict_caption_from_path(img_path, encoder, decoder, word_to_index, index_to_word, transform, device):
    img = Image.open(img_path).convert('RGB')
    feature = extract_feature(img, encoder, transform, device)
    caption = generate_caption(feature, encoder, decoder, word_to_index, index_to_word, device=device)
    return caption