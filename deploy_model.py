import torch
import pickle
from model import Encoder, DecoderTransformer
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
    embed_dim = embedding_matrix.shape[1]
    nhead = 6
    num_decoder_layers = 4
    dim_feedforward = 1024

    encoder = Encoder(embed_dim=embed_dim)
    decoder = DecoderTransformer(
        vocab_size=vocab_size,
        embedding_dim=embed_dim,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
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

def generate_caption(img, encoder, decoder, word_to_index, index_to_word, transform, device, max_len=30):
    # img: PIL Image
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(img_tensor)
        caption = [word_to_index['<START>']]
        for _ in range(max_len):
            cap_tensor = torch.tensor([caption], dtype=torch.long).to(device)
            outputs = decoder(cap_tensor, features)
            next_word_logits = outputs[0, -1, :]
            next_word = next_word_logits.argmax().item()
            caption.append(next_word)
            if next_word == word_to_index['<END>']:
                break
        words = [index_to_word[idx] for idx in caption[1:] if idx in index_to_word]
        if '<END>' in words:
            words = words[:words.index('<END>')]
        return ' '.join(words)

def predict_caption_from_path(img_path, encoder, decoder, word_to_index, index_to_word, transform, device):
    img = Image.open(img_path).convert('RGB')
    caption = generate_caption(img, encoder, decoder, word_to_index, index_to_word, transform, device)
    return caption