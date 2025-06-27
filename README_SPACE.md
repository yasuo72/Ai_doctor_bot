# Deploying MedAssist+ to a free Hugging Face Space

Follow these steps:

1. **Create a new Space** (select *Gradio* template) on your HF account.
2. **Upload the following files/folders** from this project:
   * `app.py` – interface/entry point (added by Cascade)
   * `tinyllama_rag_chatbot.py`  + helper modules
   * `rag_index.faiss` and `rag_meta.pkl`
   * `models/` directory containing `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
   * `requirements.txt`
   * `.gitignore` (optional)
3. Commit & push.  Build takes ~5 min; when it finishes your Space is live at
   `https://<space-name>.hf.space`.
4. **Flutter call example**:

```dart
final res = await http.post(
  Uri.parse('https://<space-name>.hf.space/run/predict'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({'data': [question]}),
);
final answer = jsonDecode(res.body)['data'][0];
```

That’s it—completely free hosting with HTTPS.  Note Space sleeps after ~16 h idle, first call will be ~30 seconds.
