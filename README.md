<h1 align="center"> Retrieval-Augmented Generation with Gradio and Groq API Key</h1>
<p align="center"> Natural Language Processing Project</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">

</div>

### Name : Nabilah Putri Wijaya (Universitas Semarang)
### Tech Stack : Python, Gradio, LangChain, HuggingFace Embedding, FAISS vector store

---

### 1. Analysis about how the project works
- I uploaded a pdf of the KP guide, after which the pdf will be read and the document broken into parts. Vector Storage stores these vectors in a vector database for quick search. When there is a query, the query is also converted into vectors and compared with the vectors in the database to find the most similar/relevant chunks of text. The relevant chunks of text found are then merged with the original question.  This context-enriched question is then fed to Groq's Large Language Model (LLM) (ChatGroq) to generate answers.
- I asked the question “What does the document say?” and got a good answer.
```
Jawaban: 
Dokumen tersebut adalah Pedoman Penulisan Kerja Praktek untuk mahasiswa Jurusan Teknologi Informasi di Universitas Semarang. Isi dokumen tersebut mencakup:

1. Latar belakang dan pengertian tentang kerja praktek sebagai kegiatan wajib bagi mahasiswa sebelum menempuh Tugas Akhir.
2. Tujuan kerja praktek untuk membantu mahasiswa mengenali dan menguasai profesinya sebagai calon Sarjana S1-Sistem Informasi dan S1-Teknik Informatika.
3. Informasi tentang Lampiran 8, yaitu kuesioner tingkat kepuasan DUDI terhadap mahasiswa magang yang dapat diunduh melalui aplikasi SIMA atau laman https://ftik.usm.ac.id/pages/view/Kuesioner.
4. Contoh Tanda Terima Laporan Kerja Praktek yang dapat digunakan sebagai bukti penyerahan laporan kerja praktek.
5. Kata pengantar yang menjelaskan tentang penerbitan Buku Pedoman Penulisan Kerja Praktek Edisi XVI/2024 sebagai pedoman bagi mahasiswa Jurusan Teknologi Informasi.

Dokumen tersebut juga menyebutkan bahwa buku pedoman ini berisi tentang petunjuk/cara-cara penulisan Kerja Praktek, format penulisan dan format isi, serta contoh-contoh blangko permohonan Kerja Praktek dan lain-lain.
```

### 2. Analysis about how different every model works on Retrieval-Augmented Generation

```python
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile", # Change the model in the code
        temperature=0.2
    )
```
- Model used : ```[llama-3.3-70b-versatile, deepseek-r1-distill-llama-70b, gemma2-9b-it]```

2.1 Analysis on ```llama-3.3-70b-versatile``` : 
- Question: apa tujuan kerja praktek?
- Answer: Sayangnya, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tentang tujuan kerja praktek berdasarkan konteks yang diberikan. Konteks yang diberikan hanya menyediakan beberapa lampiran dan formulir yang terkait dengan kerja praktek, tetapi tidak menjelaskan tujuan kerja praktek secara eksplisit. Jika Anda memiliki konteks lain atau informasi lebih lanjut, saya akan senang membantu. 
- Analysis: The llama-3.3-70b-versatile model behaved ideally for the RAG system in this scenario. It correctly identified the absence of relevant information in the given context and refrained from generating unsubstantiated answers. This shows that the model is able to understand the limitations of the context and prioritizes accuracy based on the source rather than trying to answer with general knowledge that may not apply.

2.2 Analysis on ```deepseek-r1-distill-llama-70b``` : 
- Question: apa tujuan kerja praktek?
- Answer: Tujuan dari kerja praktek adalah untuk memberikan pengalaman nyata dan praktis kepada mahasiswa dalam menerapkan teori dan konsep yang telah dipelajari di bangku kuliah. Dengan demikian, mahasiswa dapat mengembangkan kemampuan dan keterampilan yang dibutuhkan dalam dunia kerja, serta mempersiapkan diri untuk memasuki dunia kerja yang sebenarnya. Dalam konteks Jurusan Teknologi Informasi, kerja praktek bertujuan untuk memberikan pengalaman praktis dalam menerapkan teori dan konsep teknologi informasi, seperti pengembangan sistem, pengelolaan data, dan lain-lain. Dengan demikian, mahasiswa dapat memperoleh pengalaman nyata dan praktis dalam menerapkan pengetahuan dan keterampilan yang telah dipelajari di bangku kuliah.
- Analysis: The deepseek-r1-distill-llama-70b model provides an informative and generally relevant answer to the question “what is the purpose of the internship?”, even including a specific context for “Information Technology Department”. However, this behavior differs significantly from llama-3.3-70b-versatile which first acknowledges the absence of information in the document. This suggests that deepseek-r1-distill-llama-70b may have a lower threshold for switching to its general knowledge when the RAG context does not provide a clear answer. Although the answer itself is of good quality as general information, in a strict RAG context, it could be considered less than ideal if the goal is to obtain an answer that is only sourced from the provided document. The choice of model will depend on whether the user prefers an answer (albeit general) or an acknowledgment of the lack of specific information in the document.

2.3 Analysis on ```gemma2-9b-it``` : 
- Question: apa tujuan kerja praktek?
- Answer: Sayangnya, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tentang tujuan kerja praktek berdasarkan konteks yang diberikan. Konteks yang diberikan hanya berisi beberapa lampiran dari pedoman penulisan kerja praktek, tetapi tidak menjelaskan tujuan kerja praktek secara eksplisit. Jika Anda memiliki informasi lebih lanjut atau konteks yang lebih luas, saya akan senang membantu.
- Analysis: The gemma2-9b-it model performed very well in this RAG scenario, comparable to the much larger llama-3.3-70b-versatile in terms of its ability to recognize context limitations. The model not only states the lack of information but also provides hints about the nature of the retrieved documents (annexes of guidelines instead of explanations of purpose), indicating a good level of understanding of the retrieved content. This behavior, mainly due to its smaller size and its status as an instruction-tuned model, makes it an attractive candidate for RAG applications that require a balance between contextual accuracy, resource efficiency, and model honesty. This shows that a proper fine-tuning strategy can produce highly reliable models for specific tasks such as RAG, even with fewer parameters.

### 3. Analysis about how temperature works

```python
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2 # Change the temperature value here and analzye
    )
```

3.1 Analysis on higher temperature 0.9
- High temperatures allow the model to be more “bold” in making guesses, but can also result in critical reflection on the guesses themselves. The result can be an answer that is more dynamic and sometimes more “human” in the way it handles uncertainty (trying to guess, then admitting it was just a guess). However, this also means that the answer may be less consistent and may take longer or more convoluted to come to the same (or safer) conclusion than low temperature. There is a risk that the initial “guess” could be misleading if the user does not pay attention to the “self-correction” part.

3.2 Analysis on lower temperature 0.1
- Low temperatures result in very cautious, factual answers regarding the absence of information, and if making an inference, it is a general inference that is safe and clearly marked. The priority is accuracy based on context and avoiding excessive speculation. This answer is particularly appropriate for RAGs where contextual correctness is paramount.

### 4. How to run the project

- Clone this repository with : 

```git
git clone https://github.com/arifian853/RAG_with_GroqAPI.git
```

- Copy the ```.env.example``` file and rename it to ```.env```

```
GROQ_API_KEY=your-groq-api-key
```

- Fill the ```GROQ_API_KEY``` with your Groq API Key, find it here : https://console.groq.com/keys

- Create a Virtual Environment (Make sure Python is installed)
```
python -m venv venv
```

- Activate the Virtual Environment (windows)
```
venv\Scripts\activate
```

- Install the Required Dependencies in requirements.txt
```
pip install -r requirements.txt
```
- Running the Project (Run the main Python file app.py)
```
python app.py
```
- Once the command is executed, the terminal will display the local URL where the Gradio app is running (usually something like http://127.0.0.1:7860 or http://localhost:7860). Open this URL in your web browser.
