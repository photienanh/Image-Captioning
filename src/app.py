from flask import Flask, request, render_template
from deploy_model import load_model_and_utils, generate_caption
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__, template_folder='../templates', static_folder='../static')

encoder, decoder, word_to_index, index_to_word, transform, device = load_model_and_utils()

def image_captioning(image_bytes, encoder, decoder, word_to_index, index_to_word, transform, device):
    # Chuyển bytes thành PIL Image
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    # Sinh caption trực tiếp từ ảnh
    caption = generate_caption(img, encoder, decoder, word_to_index, index_to_word, transform, device)
    return caption

@app.route('/', methods=['GET','POST'])
def index():
    response = ""
    image_base64 = None

    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename != '':
            image_file = request.files["image"]
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode()
            response = image_captioning(
                image_bytes, encoder, decoder, word_to_index, index_to_word, transform, device
            )
    return render_template('index.html', response=response, image_base64=image_base64)