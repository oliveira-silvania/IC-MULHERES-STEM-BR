# A INCLUSÃO DE MULHERES NA CIÊNCIA BRASILEIRA EM ÁREAS DE STEM: REPOSITÓRIO DE DADOS, ANÁLISES ESTATÍSTICAS E MODELAGENS QUE IDENTIFIQUEM PADRÕES OU TENDÊNCIAS

Repositório da minha Iniciação Científica desenvolvida na **PUC Goiás**, sob orientação da **Profa. Maria José Pereira Dantas**, no período **2024/2 a 2025/1**.  

Este projeto reúne **dados, análises estatísticas, modelagens preditivas** e um **painel de Business Intelligence (BI)** sobre a participação de mulheres na ciência brasileira em áreas de STEM (Science, Technology, Engineering and Mathematics).

---

## 👤 Autoria
- **Discente:** Silvania Alves Oliveira  
- **Orientadora:** Profa. Maria José Pereira Dantas  
- **Instituição:** Pontifícia Universidade Católica de Goiás (PUC Goiás)  
- **Período:** 2024/2 – 2025/1  

---

## 🎯 Objetivos
- Construir um **repositório de dados** consolidado sobre produção científica em STEM no Brasil (2021–2023).  
- Realizar **análises estatísticas por meio de painel BI (Power BI)** para identificar padrões de participação feminina.  
- Implementar **modelagens preditivas** (ex.: Random Forest) para auxiliar na identificação de tendências.  
- Disponibilizar um **painel interativo em BI** para visualização dos resultados.  

---

## 📂 Estrutura do repositório
- **`data/`** → bases de dados utilizadas.  
  - `raw/` → dados brutos (Sucupira/CAPES).  
  - `processed/` → dados tratados para análise.  
- **`notebooks/`** → análises de predição (Python/Colab).  
- **`scripts/`** → códigos Python para modelagem e visualização.  
- **`reports/`** → relatórios e documentos finais da pesquisa.  
- **`dashboards/`** → Painel BI.  

---

## ⚙️ Tecnologias utilizadas
- **Power BI** (análises estatísticas e painel interativo)  
- **Python** (pandas, scikit-learn, matplotlib, seaborn)  
- **SQL Server** (armazenamento e consultas)  
- **GitHub** (organização e versionamento)  

---

## 📊 Painel BI
🔗 [Acesse o painel aqui](COLOQUE_O_LINK_AQUI)

---

## ▶️ Como reproduzir análises
1. Obtenha os dados (instruções em `data/README.md`).  
2. Caso queira rodar os modelos preditivos:  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows

   pip install -r requirements.txt
