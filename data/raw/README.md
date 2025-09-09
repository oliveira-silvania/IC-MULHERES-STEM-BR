# 📂 Pasta `data/raw`

Esta pasta contém os **dados brutos (raw)** utilizados no projeto de Iniciação Científica:  
> **A INCLUSÃO DE MULHERES NA CIÊNCIA BRASILEIRA EM ÁREAS DE STEM: REPOSITÓRIO DE DADOS, ANÁLISES ESTATÍSTICAS E MODELAGENS QUE IDENTIFIQUEM PADRÕES OU TENDÊNCIAS**

Os arquivos são **metadados oficiais da CAPES (2021–2024)**, extraídos do **Banco de Metadados da Plataforma Sucupira**.  
Eles descrevem a estrutura, variáveis e periodicidade de atualização dos dados homologados pelos **Programas de Pós-Graduação (PPGs)** no Brasil.

🔗 Catálogo oficial: [Metadados CAPES](https://metadados.capes.gov.br/index.php)  

---

## 📑 Arquivos disponíveis

### 1. Produção Intelectual
- **Arquivo:** `producao_intelectual.pdf`  
- **Descrição:** Metadados das produções intelectuais dos PPGs (bibliográficas, técnicas e artísticas).  
- **Fonte:** [Dados abertos CAPES – Produção Intelectual (2021–2024)](https://dadosabertos.capes.gov.br/dataset/2021-a-2024-producao-intelectual-de-pos-graduacao-stricto-sensu-no-brasil)  
- **Variáveis principais:** programa, instituição, título da produção, tipo/subtipo, área de concentração, linha de pesquisa, projeto, ISSN (quando aplicável), vínculo com TCC.  
- **Registros (2021–2023):** ~3,38 milhões  
  - **Bibliográfica:** 1,89M  
  - **Técnica:** 1,47M  
  - **Artístico-cultural:** 23k  

---

### 2. Autores da Produção Intelectual
- **Arquivo:** `autor_producao_intelectual.pdf`  
- **Descrição:** Identificação e vínculos dos autores das produções intelectuais.  
- **Fonte:** [Dados abertos CAPES – Autores da Produção Intelectual (2021–2024)](https://dadosabertos.capes.gov.br/dataset/2021-a-2024-autor-da-producao-intelectual-de-programas-de-pos-graduacao-stricto-sensu-no-brasil)  
- **Variáveis principais:** nome do autor, tipo de vínculo (docente, discente, egresso, pós-doc, externo), categoria docente, nível de titulação, área de conhecimento, país, tempo de egresso.  
- **Registros (2021–2023):** ~31,9 milhões  

---

### 3. Financiadores de Projetos
- **Arquivo:** `financiadores.pdf`  
- **Descrição:** Metadados sobre financiadores de projetos vinculados aos PPGs.  
- **Fonte:** [Dados abertos CAPES – Financiadores de Projetos (2021–2024)](https://dadosabertos.capes.gov.br/dataset/2021-a-2024-financiadores-de-projetos-dos-programas-de-pos-graduacao-stricto-sensu-no-brasil)  
- **Variáveis principais:** nome do financiador, natureza do financiamento, programa de fomento, país de origem, indicador de financiador estrangeiro, vínculo com programa/instituição.  
- **Registros (2021–2023):**  
  - **2021:** 167.959 registros → 109.808 projetos distintos, 4.424 PPGs, 431 IES  
  - **2022:** 170.027 registros → 110.939 projetos distintos, 4.362 PPGs, 432 IES  
  - **2023:** 170.624 registros → 110.743 projetos distintos, 4.407 PPGs, 434 IES  

---

### 4. Programas de Pós-Graduação
- **Arquivo:** `programas.pdf`  
- **Descrição:** Informações sobre os Programas de Pós-Graduação stricto sensu no Brasil.  
- **Fonte:** [Dados abertos CAPES – Programas de Pós-Graduação (2021–2024)](https://dadosabertos.capes.gov.br/dataset/2021-a-2024-programas-da-pos-graduacao-stricto-sensu-no-brasil)  
- **Variáveis principais:** área de conhecimento, grande área, subárea, especialidade, instituição, município, UF, região, conceito CAPES, modalidade (acadêmico/profissional), situação do programa, início do curso.  
- **Registros (2021–2023):**  
  - **2021:** 4.709 PPGs em 473 IES  
  - **2022:** 4.594 PPGs em 476 IES  
  - **2023:** 4.659 PPGs em 477 IES  

---

## 📊 Relação entre os conjuntos

```mermaid
graph TD
    A[Programas de Pós-Graduação] --> B[Produção Intelectual]
    B --> C[Autores da Produção Intelectual]
    A --> D[Projetos]
    D --> E[Financiadores]
