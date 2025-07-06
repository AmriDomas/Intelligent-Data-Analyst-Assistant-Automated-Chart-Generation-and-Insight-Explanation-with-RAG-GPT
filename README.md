# 📊 Intelligent Data Analyst Assistant Automated Chart Generation and Insight Explanation with RAG GPT

This project combines **Retrieval-Augmented Generation (RAG)** with automatic chart generation and visual data analysis using **OpenAI GPT-4o**, enabling users to extract insights from structured (CSV) and unstructured (PDF) data using **natural language**.

## 🔍 Overview

This app allows users to:

- Upload **CSV** files (structured tabular data)
- Upload **PDF** files (unstructured textual content)
- Ask **natural language questions**
- Generate **multiple charts automatically** (bar, line, pie, histogram, scatter, time series, correlation matrix)
- Get **automated analysis** written by AI logic
- Extract PDF context via **FAISS + Sentence Transformers**
- Fully powered by **OpenAI GPT-4o** LLM

---

## ⚙️ Components

| Component              | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| 🧠 **LLM (GPT-4o)**    | Generates function call(s) to plot charts + draft human answers    |
| 📊 **Auto Analysis**   | Custom Python function that analyzes chart output logically        |
| 🧾 **PDF Parser**      | Uses PyMuPDF to extract and index text                             |
| 🧠 **RAG Engine**      | FAISS + SentenceTransformer for relevant PDF context               |
| 📈 **Chart Generator** | Uses `matplotlib` and `seaborn` to plot 7 chart types              |

---

## 📁 File Structure

```bash
├── streamlit_plot.py                   # Streamlit app
├── requirements.txt
├── README.md
├── Intelligent Data Analyst Assistant.ipynb
└── data/                    # Folder to store input CSV files
   ├── US  E-commerce records 2020.csv
   └── BROSUR HIDROPONIK BAGI PEMULA.pdf
```

## 🚀 How to Run
1. Clone the repository
   ```bash
   git clone [https://github.com/AmriDomas/Intelligent-Data-Analyst-Assistant-Automated-Chart-Generation-and-Insight-Explanation-with-RAG-GPT.git]
   cd Intelligent-Data-Analyst-Assistant-Automated-Chart-Generation-and-Insight-Explanation-with-RAG-GPT
   ```
2. Create virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app
   ```bash
   streamlit run streamlit_plot.py
   ```
5. Paste your OpenAI API key in the sidebar after launching.

## 📦 Supported Chart Types

| Chart Type           | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `bar`                | Aggregated bar chart (e.g., total sales per category) |
| `line`               | Line chart across time or numeric x-axis              |
| `pie`                | Pie chart of category proportions                     |
| `scatter`            | XY scatter plot + correlation analysis                |
| `histogram`          | Distribution of numeric columns (auto-selected)       |
| `correlation_matrix` | Heatmap of correlations between numeric features      |
| `timeseries`         | Trend across date/time dimension                      |

## 💡 Example Use Case

Dataset
E-Commerce Sales with columns: City, Category, Sales, Profit

Question
"Show pie chart of sales by Category"

Output
📊 Pie chart generated from City and Category

🧠 Auto-analysis: "Category 'Technology' has contribution sum sales highest i.e 271,730.81 (37.06%)."

## 📋 Requirements

- Python 3.9+
- OpenAI API key
- See requirements.txt for details

## 🙌 Acknowledgement

This project was developed by [Linkedin](http://linkedin.com/in/muh-amri-sidiq) as a portfolio project combining:
- Data visualization
- Large Language Models
- RAG-based retrieval
- Automatic chart interpretation
