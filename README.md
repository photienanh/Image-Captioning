# IMAGE CAPTION GENERATOR

A deep learning model that automatically generates descriptive captions for images using PyTorch and ResNet50 architecture.

## 🎯 Overview

This project implements an Image Captioning system using:
- **Encoder**: ResNet50 (pretrained) for image feature extraction
- **Decoder**: Transformer with GloVe embeddings for caption generation
- **Framework**: PyTorch
- **Dataset**: Flickr8k

The model can generate natural language descriptions for images, making it useful for accessibility applications, content management, and automated image tagging.

## ✨ Features

- **Deep Learning**: Uses a state-of-the-art Encoder-Decoder architecture (ResNet50 + Transformer)
- **Pre-trained Embeddings**: Incorporates GloVe word embeddings for better language understanding
- **Web Interface**: Flask-based web application for easy image upload and caption generation
- **Flexible Input**: Supports various image formats (JPEG, PNG, etc.)

## 🎯 Demo

### Live Demo
**Try it now**: [Image Caption Generator](http://54.151.252.89:5000/)

Simply upload any image and get an AI-generated caption instantly!

### Sample Results

![Sample](https://cdn.anh.moe/f/0fI8Zr.png)



## 💻 Usage

### How to Use
1. **Visit** [http://54.151.252.89:5000/](http://54.151.252.89:5000/)
2. **Upload** your image (JPEG, PNG, etc.)
3. **Wait** for processing (usually 1-2 seconds)
4. **Get** your AI-generated caption!

**No installation required** - just upload and try!
