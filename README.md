**Cebuano Doctor**

Cebuano Doctor is a local AI healthcare chatbot that accepts healthcare
questions in Cebuano and returns medical responses in Cebuano.

**How It Works**
1. The user enters a healthcare question in Cebuano.
2. Gemma 4 translates the question from Cebuano to English.
3. MedGemma generates a medical response in English.
4. Gemma 4 translates the response back to Cebuano.
5. The Cebuano response is shown to the user.

**Tools Used**
- Python
- Ollama
- Gemma 4
- MedGemma

**Evaluation**
The chatbot was tested using five healthcare prompts provided by other
people. We evaluated:
- Cebuano-to-English translation accuracy
- Quality of the medical response
- English-to-Cebuano translation accuracy
- Performance with simple, complex, and local Cebuano terms

**Findings**
- Most simple Cebuano queries were translated correctly, but some local or
context-specific words were misunderstood. Some translations also
sounded overly formal or closer to Tagalog than everyday Cebuano.
- MedGemma generally produced relevant medical responses when the
translated input was correct. However, translation errors could affect
the medical response. For example, the Cebuano word "piang," meaning
sprain in the test prompt, resulted in a response about fungal
infection.
- We also observed that different translation tools and language models
can produce different translations for the same Cebuano words.

**What We Learned**
- Different AI models can produce different responses and translations.
- Local LLMs can perform well for many tasks, but may have more difficulty
with complex reasoning and regional language compared with larger online
models.
- The activity also showed that a multi-model system depends on every step
of the pipeline. Even if the medical model gives a good response, an
incorrect translation of the original query can lead to an incorrect
final answer.
