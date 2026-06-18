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
```

This allows iterative personality improvements through future fine-tuning.

## Model Setup

The repository does not include the base language model because GitHub file size limits make it impractical to distribute large model weights through the repository.

The original version of Fran used **Llama 3.1 8B** as the base model, which was stored in:

```text
D:/Fran/brain/base_model
```

The model files are intentionally excluded from this repository.

Fran is not tied to Llama 3.1 8B. You can replace it with any compatible chat or instruction-tuned model supported by Hugging Face Transformers by updating the model path in the configuration.

When replacing the model, ensure that:

- The tokenizer matches the selected model
- The model supports chat-style prompting
- Available VRAM is sufficient for the chosen model
- Generation settings are adjusted if necessary

## Current Capabilities

- Voice-based conversations
- Twitch event reactions
- Character-driven responses
- Context-aware chat history
- Real-time speech generation
- Training data collection

## Future Plans

- Improved memory systems
- Additional Twitch interactions
- Automated fine-tuning pipeline
- Enhanced personality training
- Multi-character support

## Repository

Source code:

https://github.com/ShovitXD/Fran

## Disclaimer

Fran is a personal learning and experimentation project focused on conversational AI, speech systems, real-time event processing, and LLM fine-tuning workflows.
