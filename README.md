# A INCLUSÃO DE MULHERES NA CIÊNCIA BRASILEIRA EM ÁREAS DE STEM: REPOSITÓRIO DE DADOS, ANÁLISES ESTATÍSTICAS E MODELAGENS QUE IDENTIFIQUEM PADRÕES OU TENDÊNCIAS

Repositório da minha Iniciação Científica desenvolvida na **PUC Goiás**, sob orientação da **Profa. Maria José Pereira Dantas**, no período **2024/2 a 2025/1**.  

Este projeto reúne **dados, análises estatísticas, modelagens preditivas** e um **painel de Business Intelligence (BI)** sobre a participação de mulheres na ciência brasileira em áreas de STEM (Science, Technology, Engineering and Mathematics).

---

## 🎯 Objetivos
- Construir um **repositório de dados** consolidado sobre produção científica em STEM no Brasil (2021–2023).  
- Realizar **análises estatísticas** e identificar padrões de participação feminina.  
- Implementar **modelagens preditivas** (ex.: Random Forest) para auxiliar na identificação de tendências.  
- Disponibilizar um **painel interativo em BI** para visualização dos resultados.  

---

## 📂 Estrutura do repositório
- **`data/`** → bases de dados utilizadas.  
  - `raw/` → dados brutos (Sucupira/CAPES).  
  - `processed/` → dados tratados para análise.  
- **`notebooks/`** → análises exploratórias e estatísticas em Jupyter/Colab.  
- **`scripts/`** → códigos Python para processamento, modelagem e visualização.  
- **`reports/`** → relatórios e documentos finais da pesquisa.  
- **`dashboards/`** → arquivos do painel BI (Power BI `.pbix`) e exportações.  

---

## ⚙️ Tecnologias utilizadas
- **Python** (pandas, numpy, scikit-learn, matplotlib, seaborn)  
- **SQL Server** (armazenamento e consultas)  
- **Power BI** (painel interativo)  
- **GitHub** (organização e versionamento)  

---

## 📊 Painel BI
🔗 [Acesse o painel aqui](COLOQUE_O_LINK_AQUI)

---

## ▶️ Como reproduzir análises
1. Obtenha os dados (instruções em `data/README.md`).  
2. Crie o ambiente virtual e instale dependências:  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows

   pip install -r requirements.txt
