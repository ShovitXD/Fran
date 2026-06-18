# Fran

Fran is a local AI-powered Twitch co-host designed for real-time voice interaction, Twitch chat engagement, and continuous personality improvement through dataset collection and fine-tuning.

The project combines speech recognition, local LLM inference, Twitch event integration, and text-to-speech into a single conversational system.

## Features

- Real-time voice conversations using push-to-talk
- Local LLM inference with 4-bit quantization
- Speech-to-text using Faster-Whisper
- Text-to-speech using Gemini TTS
- Twitch EventSub integration
- Awareness of follows, subscriptions, cheers, and channel point redemptions
- Conversation history and contextual responses
- Dataset capture pipeline for future fine-tuning
- Low VRAM optimized inference

## Architecture

Voice Input  
→ Faster-Whisper  
→ Local LLM  
→ Gemini TTS  
→ Audio Output

Twitch Events  
→ Event Listener  
→ Local LLM  
→ Gemini TTS  
→ Audio Output

## Technologies Used

### Speech Recognition

- Faster-Whisper
- Whisper Large-v3-Turbo
- CUDA FP16 inference

### Language Model

- Hugging Face Transformers
- BitsAndBytes
- 4-bit NF4 Quantization
- Unsloth (fine-tuned model support)

### Voice Synthesis

- Google Gemini TTS
- Leda Voice

### Twitch Integration

- TwitchAPI
- EventSub WebSockets

### Core Stack

- Python
- PyTorch
- NumPy
- SoundDevice
- AsyncIO

## Dataset Collection

Fran includes a built-in dataset capture system.

Selected conversations can be saved as structured JSONL training samples:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello Fran"
    },
    {
      "role": "assistant",
      "content": "Hey. You're back already?"
    }
  ]
}
