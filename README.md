<h1 align="center"> Retrieval-Augmented Generation with Gradio and Groq API Key</h1>
<p align="center"> Natural Language Processing Project</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">

</div>

### Name : Nabilah Putri Wijaya (Universitas Semarang)
### Tech Stack : Python, Gradio, LangChain, HuggingFace Embedding, FAISS vector store

---

### 1. Analysis about how the project works
- Saya mengunggah sebuah pdf dari panduan KP, setelah itu pdf tersebut akan dibaca dan dokumen dipecah menjadi beberapa bagian. Penyimpanan Vektor menyimpan vektor-vektor ini dalam basis data vektor untuk pencarian cepat. Ketika ada kueri, kueri juga dikonversi menjadi vektor dan dibandingkan dengan vektor dalam database untuk menemukan potongan teks yang paling mirip/relevan. Potongan teks yang relevan yang ditemukan kemudian digabungkan dengan pertanyaan asli.  Pertanyaan yang diperkaya dengan konteks ini kemudian diumpankan ke Large Language Model (LLM) Groq (ChatGroq) untuk menghasilkan jawaban.
- Saya mengajukan pertanyaan “Apa isi dari dokumen tersebut?” dan mendapatkan jawaban yang bagus.
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
- Analysis: Model llama-3.3-70b-versatile berperilaku secara ideal untuk sistem RAG dalam skenario ini. Ia dengan benar mengidentifikasi ketiadaan informasi yang relevan dalam konteks yang diberikan dan menahan diri untuk tidak menghasilkan jawaban yang tidak berdasar. Ini menunjukkan bahwa model tersebut mampu memahami batasan konteks dan lebih memprioritaskan akurasi berdasarkan sumber daripada mencoba menjawab dengan pengetahuan umum yang mungkin tidak berlaku.

2.2 Analysis on ```deepseek-r1-distill-llama-70b``` : 
- Question: apa tujuan kerja praktek?
- Answer: Tujuan dari kerja praktek adalah untuk memberikan pengalaman nyata dan praktis kepada mahasiswa dalam menerapkan teori dan konsep yang telah dipelajari di bangku kuliah. Dengan demikian, mahasiswa dapat mengembangkan kemampuan dan keterampilan yang dibutuhkan dalam dunia kerja, serta mempersiapkan diri untuk memasuki dunia kerja yang sebenarnya. Dalam konteks Jurusan Teknologi Informasi, kerja praktek bertujuan untuk memberikan pengalaman praktis dalam menerapkan teori dan konsep teknologi informasi, seperti pengembangan sistem, pengelolaan data, dan lain-lain. Dengan demikian, mahasiswa dapat memperoleh pengalaman nyata dan praktis dalam menerapkan pengetahuan dan keterampilan yang telah dipelajari di bangku kuliah.
- Analysis: Model deepseek-r1-distill-llama-70b memberikan jawaban yang informatif dan relevan secara umum terhadap pertanyaan "apa tujuan kerja praktek?", bahkan menyertakan konteks spesifik untuk "Jurusan Teknologi Informasi". Namun, perilaku ini berbeda signifikan dari llama-3.3-70b-versatile yang lebih dulu mengakui ketiadaan informasi dalam dokumen. Ini menunjukkan bahwa deepseek-r1-distill-llama-70b mungkin memiliki ambang batas yang lebih rendah untuk beralih ke pengetahuan umumnya ketika konteks RAG tidak memberikan jawaban yang jelas. Meskipun jawabannya sendiri berkualitas baik sebagai informasi umum, dalam konteks RAG yang ketat, ini bisa dianggap kurang ideal jika tujuannya adalah untuk mendapatkan jawaban yang hanya bersumber dari dokumen yang disediakan. Pemilihan model akan bergantung pada apakah pengguna lebih memilih jawaban (meskipun umum) atau pengakuan atas kurangnya informasi spesifik dalam dokumen.

2.3 Analysis on ```gemma2-9b-it``` : 
- Question: apa tujuan kerja praktek?
- Answer: Sayangnya, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tentang tujuan kerja praktek berdasarkan konteks yang diberikan. Konteks yang diberikan hanya berisi beberapa lampiran dari pedoman penulisan kerja praktek, tetapi tidak menjelaskan tujuan kerja praktek secara eksplisit. Jika Anda memiliki informasi lebih lanjut atau konteks yang lebih luas, saya akan senang membantu.
- Analysis: Model gemma2-9b-it menunjukkan kinerja yang sangat baik dalam skenario RAG ini, sebanding dengan llama-3.3-70b-versatile yang jauh lebih besar dalam hal kemampuannya untuk mengakui keterbatasan konteks. Model ini tidak hanya menyatakan kekurangan informasi tetapi juga memberikan petunjuk tentang sifat dokumen yang diambil (lampiran pedoman, bukan penjelasan tujuan), yang menunjukkan tingkat pemahaman yang baik terhadap konten yang diambil. Perilaku ini, terutama karena ukurannya yang lebih kecil dan statusnya sebagai model yang di-instruction-tuned, menjadikannya kandidat yang menarik untuk aplikasi RAG yang membutuhkan keseimbangan antara akurasi kontekstual, efisiensi sumber daya, dan kejujuran model. Ini menunjukkan bahwa strategi fine-tuning yang tepat dapat menghasilkan model yang sangat andal untuk tugas-tugas spesifik seperti RAG, bahkan dengan parameter yang lebih sedikit.

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
- Temperatur tinggi memungkinkan model untuk menjadi lebih "berani" dalam membuat tebakan, tetapi juga bisa menghasilkan refleksi kritis terhadap tebakan itu sendiri. Hasilnya bisa menjadi jawaban yang lebih dinamis dan terkadang lebih "manusiawi" dalam cara ia menangani ketidakpastian (mencoba menebak, lalu mengakui itu hanya tebakan). Namun, ini juga berarti jawabannya mungkin kurang konsisten dan bisa jadi lebih panjang atau berbelit-belit untuk sampai pada kesimpulan yang sama (atau lebih aman) dibandingkan temperatur rendah. Ada risiko bahwa "tebakan" awal bisa menyesatkan jika pengguna tidak memperhatikan bagian "self-correction".

3.2 Analysis on lower temperature 0.1
- Temperatur rendah menghasilkan jawaban yang sangat berhati-hati, faktual mengenai ketiadaan informasi, dan jika membuat inferensi, itu adalah inferensi umum yang aman dan ditandai dengan jelas. Prioritasnya adalah akurasi berdasarkan konteks dan menghindari spekulasi berlebih. Jawaban ini sangat sesuai untuk RAG di mana kebenaran kontekstual sangat dijaga.

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
