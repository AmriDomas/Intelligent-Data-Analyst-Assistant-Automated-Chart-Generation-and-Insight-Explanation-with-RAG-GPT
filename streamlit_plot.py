import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import faiss
import numpy as np
import fitz  # PyMuPDF
import json
import re
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# ----------- FAISS + RAG ------------
def build_faiss_index_cosine(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve(query, index, df, top_k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec)
    query_vec = query_vec.astype('float32')
    scores, indices = index.search(query_vec, top_k)
    return df.iloc[indices[0]]

def generate_function_and_answer(query, context_schema, api_key):
    import openai
    openai.api_key = api_key

    system_message = """
Kamu adalah asisten cerdas untuk analisis data berbasis visual.

Tugasmu:
1. Berdasarkan pertanyaan dan skema data yang diberikan, buat SATU atau LEBIH function call untuk visualisasi data (dalam JSON list).
2. Gunakan hanya kolom yang tersedia dalam skema. Jangan menggunakan kolom di luar daftar tersebut.
3. Jika tidak ada kombinasi kolom yang cocok untuk membuat grafik, berikan hanya jawaban teks.
4. Setelah visualisasi dibuat, berikan jawaban manusia yang menyimpulkan isi grafik sesuai hasil dari plot (bukan sekadar menyebutkan nilai angka).

📊 Jenis plot yang bisa kamu hasilkan:
- `bar`: data kategorikal vs numerik (jumlah, rata-rata)
- `line`: tren data berurutan
- `scatter`: korelasi dua variabel numerik
- `pie`: proporsi kategori
- `timeseries`: time-based trends (gunakan jika ada kolom tanggal)
- `histogram`: distribusi numerik (tanpa 'y' pun bisa)
- `correlation_matrix`: korelasi antar semua kolom numerik

Format Output:
Function:
[
  {"function": "plot_chart", "args": {"plot_type": "bar", "x": "wilayah", "y": "penjualan", "agg": "sum"}},
  {"function": "plot_chart", "args": {"plot_type": "timeseries", "x": "tanggal", "y": "profit", "agg": "mean", "filters": {"segmen": "VIP"}}}
]

Jawaban:
Toko di wilayah Jakarta memiliki penjualan tertinggi, sedangkan margin profit paling stabil ditemukan di segmen VIP pada kuartal terakhir.
"""

    user_message = f"""Pertanyaan: {query}

Skema Kolom Dataset:
{context_schema}

Jawaban hanya boleh berdasarkan kolom-kolom di atas. Jika pertanyaan tidak cocok dengan kolom tersebut, berikan jawaban teks tanpa grafik."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": user_message.strip()}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message["content"]



def parse_function_response(response_text):
    match = re.search(r'Function:\s*(\[.*?\])\s*Jawaban:', response_text, re.DOTALL)
    answer_match = re.search(r'Jawaban:\s*(.*)', response_text, re.DOTALL)

    function_calls = []
    if match:
        try:
            function_calls = json.loads(match.group(1))
        except json.JSONDecodeError:
            function_calls = []

    answer_text = answer_match.group(1).strip() if answer_match else ""
    return function_calls, answer_text

def extract_text_from_pdfs(files):
    data = []
    for file in files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                data.append({"source": file.name, "page": page_num + 1, "text": text.strip()})
    return pd.DataFrame(data)

def transform_csv(df, selected_columns, source_name):
    df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
    df["source"] = source_name
    df["page"] = None
    return df[["source", "page", "text"]]

# Auto detect columns info for CSV
def auto_detect_columns_info(df):
    context_info = "Schema kolom dari data:\n"
    for col in df.columns:
        try:
            sample_val = df[col].dropna().iloc[0]
            is_date = False
            # Coba parse datetime
            try:
                pd.to_datetime(sample_val)
                is_date = True
            except:
                pass

            dtype = "datetime" if is_date else str(df[col].dtype)
        except:
            dtype = str(df[col].dtype)

        # Ambil 3 sample unik (stringify agar tidak error dengan tipe aneh)
        sample = list(map(str, df[col].dropna().unique()[:3]))
        context_info += f"- {col} (type: {dtype}, sample: {sample})\n"
    return context_info


# -------------- Plot Executor --------------
def execute_plot(args, df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_type = args.get("plot_type")
    x = args.get("x")
    y = args.get("y")
    agg = args.get("agg", "sum")
    filters = args.get("filters", {})

    if not plot_type or plot_type not in ["bar", "line", "scatter", "pie", "timeseries", "histogram", "correlation_matrix"]:
        st.warning(f"📛 Type plot '{plot_type}' not recognized or not yet supported.")
        return

    df_filtered = df.copy()

    # --- Konversi tanggal jika ada kolom tanggal
    for col in df_filtered.columns:
        if "date" in col.lower() or "tanggal" in col.lower():
            try:
                df_filtered[col] = pd.to_datetime(df_filtered[col], errors='coerce')
            except:
                pass

    # --- Terapkan filters
    for col, val in filters.items():
        if col not in df_filtered.columns:
            continue
        if isinstance(val, list):
            df_filtered = df_filtered[df_filtered[col].isin(val)]
        elif isinstance(val, str) and val.isdigit() and 'date' in col.lower():
            df_filtered = df_filtered[df_filtered[col].dt.year == int(val)]
        else:
            df_filtered = df_filtered[df_filtered[col] == val]

    # --- Tangani Data Kosong
    if df_filtered.empty:
        st.warning("📭 No data matching the filter.")
        return

    # --- Agregasi (jika diperlukan)
    if plot_type in ["bar", "line", "pie", "timeseries"] and x and y:
        if agg == "sum":
            grouped = df_filtered.groupby(x)[y].sum().reset_index()
        elif agg == "mean":
            grouped = df_filtered.groupby(x)[y].mean().reset_index()
        elif agg == "median":
            grouped = df_filtered.groupby(x)[y].median().reset_index()
        elif agg == "count":
            grouped = df_filtered.groupby(x)[y].count().reset_index()
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")
    else:
        grouped = df_filtered

    # --- Plot
    plt.figure(figsize=(10, 5))

    if plot_type == "bar":
        plt.bar(grouped[x], grouped[y])
        plt.title(f"Bar Chart of {y} by {x} ({agg})")
        plt.xticks(rotation=45)

    elif plot_type == "line":
        plt.plot(grouped[x], grouped[y], marker='o')
        plt.title(f"Line Chart of {y} by {x} ({agg})")

    elif plot_type == "scatter":
        plt.scatter(grouped[x], grouped[y])
        plt.title(f"Scatter Plot: {y} vs {x}")

    elif plot_type == "pie":
        plt.pie(grouped[y], labels=grouped[x], autopct='%1.1f%%')
        plt.title(f"Pie Chart of {y} by {x} ({agg})")

    elif plot_type == "timeseries":
        grouped[x] = pd.to_datetime(grouped[x], errors='coerce')
        grouped = grouped.sort_values(x)
        plt.plot(grouped[x], grouped[y])
        plt.title(f"Time Series: {y} over {x} ({agg})")

    elif plot_type == "histogram":
       numeric_cols = df_filtered.select_dtypes(include='number').columns.tolist()

       if y:
           if isinstance(y, list):  # bisa dari LLM sebagai list
               cols_to_plot = [col for col in y if col in numeric_cols]
           else:
               cols_to_plot = [y] if y in numeric_cols else []
       else:
           cols_to_plot = numeric_cols

       if not cols_to_plot:
           st.error("📛 No numeric column can be used for histogram.")
           return

       for col in cols_to_plot:
           sns.histplot(df_filtered[col], kde=True)
           plt.title(f"Histogram of {col}")
           plt.xlabel(col)
           plt.tight_layout()
           st.pyplot(plt)
           plt.clf()
       return


    elif plot_type == "correlation_matrix":
        numeric_df = df_filtered.select_dtypes(include='number')
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")

    else:
        st.error(f"Type plot '{plot_type}' not recognized or not yet supported.")
        return

    plt.tight_layout()
    st.pyplot(plt)

# -------------- Auto Answer Generator --------------
def generate_auto_answer(df, args):
    try:
        plot_type = args.get("plot_type")
        x = args.get("x")
        y = args.get("y")
        agg = args.get("agg", "sum")
        filters = args.get("filters", {})

        # Filter data
        for k, v in filters.items():
            if k in df.columns:
                df = df[df[k] == v]

        if df.empty:
            return "Empty data after filtering."

        # Bar, Line, Timeseries
        if plot_type in ["bar", "line", "timeseries"]:
            if x not in df.columns or y not in df.columns:
                return f"columns {x} or {y} Not found in the data."

            grouped = df.groupby(x)[y].agg(agg).reset_index()
            grouped = grouped.sort_values(by=y, ascending=False)
            top = grouped.iloc[0]
            percent = round(100 * top[y] / grouped[y].sum(), 2)

            return f"Category '{top[x]}' have {agg} {y.lower()} highest, i.e {top[y]:,.2f} ({percent}%) compared to other categories."

        # Scatter plot
        elif plot_type == "scatter":
            if x not in df.columns or y not in df.columns:
                return f"columns {x} or {y} Not found in the data."

            corr = df[[x, y]].corr().iloc[0, 1]
            return f"There is a correlation of {corr:.2f} between '{x}' and '{y}'."

        # Pie chart
        elif plot_type == "pie":
            if x not in df.columns or y not in df.columns:
                return f"columns {x} or {y} Not found in the data."

            pie_group = df.groupby(x)[y].agg(agg).reset_index()
            top = pie_group.sort_values(by=y, ascending=False).iloc[0]
            percent = round(100 * top[y] / pie_group[y].sum(), 2)
            return f"Category '{top[x]}' has contribution {agg} {y.lower()} highest i.e {top[y]:,.2f} ({percent}%)."

        # Correlation Matrix
        elif plot_type == "correlation_matrix":
            numeric_cols = df.select_dtypes(include='number').columns
            corr_matrix = df[numeric_cols].corr()
            high_corr = corr_matrix.where(~np.eye(len(corr_matrix),dtype=bool)).abs().stack().sort_values(ascending=False)
            if not high_corr.empty:
                top_pair = high_corr.idxmax()
                value = high_corr.max()
                return f"The highest correlation is between '{top_pair[0]}' and '{top_pair[1]}', with value {value:.2f}."
            else:
                return "No significant correlation found between numeric columns."

        # Histogram
        elif plot_type == "histogram":
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            return f"The histogram will show the distribution of the following numerical columns: {', '.join(numeric_cols)}."

        else:
            return f"Type plot '{plot_type}' not recognized."

    except Exception as e:
        return f"Fails to generate auto-answer: {e}"

# ---------------- UI ----------------
st.title("📄🔍 Intelligent Data Analyst Assistant: Automated Chart Generation and Insight Explanation with RAG & GPT")

st.sidebar.markdown("### 📥 Upload Files")
csv_files = st.sidebar.file_uploader("Upload CSV File(s)", type='csv', accept_multiple_files=True)
pdf_files = st.sidebar.file_uploader("Upload PDF File(s)", type='pdf', accept_multiple_files=True)
input_api_key = st.sidebar.text_input("🔑 Input OpenAI API Key", type='password')
button_api = st.sidebar.button("Activate API Key")
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if input_api_key and button_api:
    st.session_state.api_key = input_api_key
    st.sidebar.success("API Key Activated!")

final_df = pd.DataFrame(columns=["source", "page", "text"])
all_csv_dfs = {}

if csv_files:
    for file in csv_files:
        try:
            df_csv = pd.read_csv(file, encoding='cp1252')
        except:
            df_csv = pd.read_csv(file)
        all_csv_dfs[file.name] = df_csv
        st.subheader(f"📄 Select Columns from {file.name}")
        selected_columns = st.multiselect(
            f"Select columns for: {file.name}",
            options=df_csv.columns.tolist(),
            default=df_csv.columns.tolist(),
            key=file.name
        )
        if selected_columns:
            transformed = transform_csv(df_csv, selected_columns, file.name)
            final_df = pd.concat([final_df, transformed], ignore_index=True)

if pdf_files:
    st.subheader("📑 Preview of Extracted PDF Content")
    df_pdf = extract_text_from_pdfs(pdf_files)
    if not df_pdf.empty:
        st.dataframe(df_pdf[["source", "page", "text"]].head())
        final_df = pd.concat([final_df, df_pdf], ignore_index=True)

if not final_df.empty:
    query = st.text_input("🧠 Enter your question", placeholder="Contoh: Buatkan diagram pie untuk penjualan berdasarkan sub-kategori.")
    run_query = st.button("Answer the Question")

    if run_query:
        if not st.session_state.api_key:
            st.warning("⚠️ Please activate your API Key first.")
        else:
            file_df = next(iter(all_csv_dfs.values())) if all_csv_dfs else pd.DataFrame()
            context_schema = auto_detect_columns_info(file_df)

            with st.spinner("🔎 Searching for relevant context..."):
                index, _ = build_faiss_index_cosine(final_df["text"].tolist())
                results = retrieve(query, index, final_df)
                context = "\n\n".join(results["text"].tolist()) + "\n\nSchema:\n" + context_schema

            with st.spinner("✍️ Generating instruction and answer..."):
                response_text = generate_function_and_answer(query, context, st.session_state.api_key)
                function_calls, human_answer = parse_function_response(response_text)

            file_df = next(iter(all_csv_dfs.values())) if all_csv_dfs else pd.DataFrame()
            auto_answer = ""
            source_is_pdf = all_csv_dfs == {}  # Tidak ada file CSV = hanya PDF

            if function_calls:
                for i, call in enumerate(function_calls):
                    args = call["args"]
                    st.markdown(f"#### 📊 Chart {i+1} ({args.get('y', '')}):")
                    try:
                        execute_plot(args, file_df)
                        auto_text = generate_auto_answer(file_df.copy(), args)
                        auto_answer += f"- {auto_text}\n"
                        with st.container():
                            st.markdown(f"🧠 **Analisis Otomatis:** {auto_text}")
                    except Exception as e:
                        st.error(f"❌ Error in chart {i+1}: {e}")

                # Tampilkan satu jawaban tergantung sumber data
                st.subheader("🧾 Answer:")
                if source_is_pdf:
                    st.success(human_answer)
                else:
                    st.success(auto_answer.strip())

            else:
                try:
                    st.subheader("🧾 Answer:")
                    st.success(human_answer)
                    st.markdown("### 📌 Source(s):")
                    st.write(results[["source", "page"]])
                except:
                    st.warning("❗ Jawaban hanya berupa teks, tidak ada function call yang valid.")

            st.markdown("### 📌 Source(s):")
            st.write(results[["source", "page"]])
else:
    st.info("⬅️ Please upload at least one CSV or PDF file.")

